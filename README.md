"""
MCP (Model Context Protocol) endpoint
Compatível com Claude, GPT-4, Gemini e qualquer LLM que suporte MCP over HTTP
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Optional, List
import json

from core.auth import get_client
from core.database import log_usage

router = APIRouter()

# ─── Schema MCP ───────────────────────────────────────────────────────────────

MCP_TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Busca semântica na base de conhecimento do cliente. "
            "Use para responder perguntas com base nos documentos ingeridos."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Pergunta ou texto para buscar na base de conhecimento",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de resultados retornados (padrão: 5)",
                    "default": 5,
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace específico para buscar (opcional)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ingest_text",
        "description": (
            "Ingere texto diretamente na base de conhecimento. "
            "Use para adicionar informações dinâmicas como FAQs, perfis ou notas."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Texto a ser ingerido",
                },
                "source_name": {
                    "type": "string",
                    "description": "Nome/label da fonte (ex: 'faq_pricing', 'icp_perfil')",
                },
                "namespace": {
                    "type": "string",
                    "description": "Namespace de destino (opcional, usa o padrão do cliente)",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "get_index_stats",
        "description": "Retorna estatísticas do índice Pinecone: total de vetores, namespaces, dimensão.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _search(query: str, top_k: int, namespace: str, client: dict) -> dict:
    import httpx
    host = client["pinecone_host"]
    if not host.startswith("http"):
        host = f"https://{host}"

    # Embed
    async with httpx.AsyncClient(timeout=30) as http:
        embed_resp = await http.post(
            "https://api.pinecone.io/embed",
            headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"},
            json={
                "model": "multilingual-e5-large",
                "inputs": [{"text": query}],
                "parameters": {"input_type": "query", "truncate": "END"},
            },
        )
        if embed_resp.status_code != 200:
            return {"error": f"Embed failed: {embed_resp.text}"}

        vector = embed_resp.json()["data"][0]["values"]

        # Query
        query_resp = await http.post(
            f"{host}/query",
            headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"},
            json={
                "vector": vector,
                "topK": top_k,
                "includeMetadata": True,
                "includeValues": False,
                "namespace": namespace,
            },
        )
        if query_resp.status_code != 200:
            return {"error": f"Query failed: {query_resp.text}"}

    matches = query_resp.json().get("matches", [])
    return {
        "total_matches": len(matches),
        "results": [
            {
                "score": round(m["score"], 4),
                "text": m.get("metadata", {}).get("text", ""),
                "source": m.get("metadata", {}).get("source", ""),
            }
            for m in matches
        ],
    }

async def _ingest_text(text: str, source_name: str, namespace: str, client: dict) -> dict:
    import httpx, hashlib, uuid

    chunks = []
    chunk_size, overlap = 800, 150
    start = 0
    clean = " ".join(text.split())
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    async with httpx.AsyncClient(timeout=60) as http:
        embed_resp = await http.post(
            "https://api.pinecone.io/embed",
            headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"},
            json={
                "model": "multilingual-e5-large",
                "inputs": [{"text": c} for c in chunks],
                "parameters": {"input_type": "passage", "truncate": "END"},
            },
        )
        if embed_resp.status_code != 200:
            return {"error": f"Embed failed: {embed_resp.text}"}

        embeddings = [item["values"] for item in embed_resp.json()["data"]]
        doc_id = hashlib.md5(source_name.encode()).hexdigest()[:8]
        vectors = [
            {
                "id": f"{doc_id}_chunk_{i}_{uuid.uuid4().hex[:4]}",
                "values": emb,
                "metadata": {"source": source_name, "chunk_index": i, "text": c, "client_id": client["id"]},
            }
            for i, (c, emb) in enumerate(zip(chunks, embeddings))
        ]

        host = client["pinecone_host"]
        if not host.startswith("http"):
            host = f"https://{host}"

        upsert_resp = await http.post(
            f"{host}/vectors/upsert",
            headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"},
            json={"vectors": vectors, "namespace": namespace},
        )
        if upsert_resp.status_code != 200:
            return {"error": f"Upsert failed: {upsert_resp.text}"}

    return {"status": "success", "chunks_ingested": len(chunks), "source": source_name}

# ─── MCP Endpoints ────────────────────────────────────────────────────────────

@router.get("")
async def mcp_discover(client: dict = Depends(get_client)):
    """MCP discovery endpoint — retorna tools disponíveis"""
    return {
        "schema_version": "v1",
        "name": f"pinecone-kb-{client['name'].lower().replace(' ', '-')}",
        "description": f"Base de conhecimento Pinecone para {client['name']}",
        "tools": MCP_TOOLS,
    }

@router.post("/call")
async def mcp_call(request: Request, client: dict = Depends(get_client)):
    """
    MCP tool call endpoint.
    Payload esperado: { "name": "tool_name", "input": { ... } }
    """
    body = await request.json()
    tool_name = body.get("name")
    tool_input = body.get("input", {})
    namespace = tool_input.get("namespace") or client["namespace"]

    log_usage(client["id"], f"/mcp/call:{tool_name}")

    if tool_name == "search_knowledge_base":
        query = tool_input.get("query", "")
        top_k = int(tool_input.get("top_k", 5))
        result = await _search(query, top_k, namespace, client)
        return {"type": "tool_result", "content": result}

    elif tool_name == "ingest_text":
        text = tool_input.get("text", "")
        source_name = tool_input.get("source_name", "mcp_input")
        result = await _ingest_text(text, source_name, namespace, client)
        return {"type": "tool_result", "content": result}

    elif tool_name == "get_index_stats":
        import httpx
        host = client["pinecone_host"]
        if not host.startswith("http"):
            host = f"https://{host}"
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(
                f"{host}/describe_index_stats",
                headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"},
                json={},
            )
        return {"type": "tool_result", "content": resp.json()}

    else:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' não encontrada")
