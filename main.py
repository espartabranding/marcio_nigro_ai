"""
MCP Pinecone Server - arquivo único
Embeddings via OpenAI text-embedding-3-small (512 dims)
Parsing via LlamaParse (multimodal - PDFs, slides, imagens, tabelas)
"""

from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, List
import sqlite3, secrets, uuid, hashlib, httpx, io, os, json, asyncio, time

DB_PATH = os.getenv("DB_PATH", "/tmp/mcp_server.db")
ADMIN_KEY_ENV = os.getenv("ADMIN_KEY", "admin-change-me")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE, CHUNK_OVERLAP = 800, 150

# ── Database ──────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS clients (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT,
            api_key TEXT UNIQUE NOT NULL, pinecone_api_key TEXT NOT NULL,
            pinecone_host TEXT NOT NULL, namespace TEXT NOT NULL DEFAULT 'default',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')), notes TEXT
        );
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT NOT NULL,
            endpoint TEXT NOT NULL, tokens_used INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS ingestion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT NOT NULL,
            filename TEXT NOT NULL, chunks INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()

def log_usage(client_id, endpoint):
    conn = get_conn()
    conn.execute("INSERT INTO api_usage (client_id, endpoint) VALUES (?, ?)", (client_id, endpoint))
    conn.commit(); conn.close()

def log_ingestion(client_id, filename, chunks, status):
    conn = get_conn()
    conn.execute("INSERT INTO ingestion_log (client_id, filename, chunks, status) VALUES (?, ?, ?, ?)",
                 (client_id, filename, chunks, status))
    conn.commit(); conn.close()

# ── Auth ──────────────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_client(x_api_key: str = Depends(api_key_header)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header obrigatório")
    conn = get_conn()
    row = conn.execute("SELECT * FROM clients WHERE api_key = ? AND active = 1", (x_api_key,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=403, detail="API Key inválida ou cliente inativo")
    return dict(row)

def get_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")):
    if x_admin_key != ADMIN_KEY_ENV:
        raise HTTPException(status_code=403, detail="Admin key inválida")
    return True

# ── LlamaParse ────────────────────────────────────────────────────────────────

async def parse_with_llama(content: bytes, filename: str) -> str:
    """Envia documento para LlamaParse e retorna texto extraído com interpretação multimodal"""
    async with httpx.AsyncClient(timeout=120) as client:
        # Upload do arquivo
        upload = await client.post(
            "https://api.cloud.llamaindex.ai/api/parsing/upload",
            headers={"Authorization": f"Bearer {LLAMA_API_KEY}"},
            files={"file": (filename, content, "application/octet-stream")},
            data={
                "language": "pt",
                "parsing_instruction": (
                    "Extraia todo o conteúdo deste documento de forma detalhada. "
                    "Para gráficos e tabelas, descreva os dados numericamente. "
                    "Para slides, extraia título, bullets e qualquer dado visual. "
                    "Mantenha a estrutura lógica do conteúdo."
                ),
            }
        )
        if upload.status_code != 200:
            raise HTTPException(status_code=502, detail=f"LlamaParse upload error: {upload.text}")

        job_id = upload.json().get("id")
        if not job_id:
            raise HTTPException(status_code=502, detail="LlamaParse não retornou job_id")

        # Polling até completar
        for _ in range(60):
            await asyncio.sleep(3)
            status_r = await client.get(
                f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}",
                headers={"Authorization": f"Bearer {LLAMA_API_KEY}"}
            )
            status_data = status_r.json()
            job_status = status_data.get("status", "")
            if job_status == "SUCCESS":
                break
            elif job_status == "ERROR":
                raise HTTPException(status_code=502, detail=f"LlamaParse job falhou: {status_data}")

        # Busca resultado em texto
        result_r = await client.get(
            f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/text",
            headers={"Authorization": f"Bearer {LLAMA_API_KEY}"}
        )
        if result_r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"LlamaParse result error: {result_r.text}")

        return result_r.json().get("text", "")

# ── Fallback text extraction ───────────────────────────────────────────────────

async def extract_text_fallback(content: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return content.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        try:
            from docx import Document
            return "\n".join(p.text for p in Document(io.BytesIO(content)).paragraphs)
        except ImportError:
            raise HTTPException(status_code=422, detail="python-docx não instalado")
    raise HTTPException(status_code=415, detail=f"Formato não suportado: {filename}")

async def extract_text(file: UploadFile) -> str:
    content = await file.read()
    name = file.filename.lower()
    # TXT e MD não precisam de LlamaParse
    if name.endswith(".txt") or name.endswith(".md"):
        return content.decode("utf-8", errors="ignore")
    # Tudo mais passa pelo LlamaParse (PDF, DOCX, PPTX, imagens)
    if LLAMA_API_KEY:
        return await parse_with_llama(content, file.filename)
    # Fallback sem LlamaParse
    return await extract_text_fallback(content, file.filename)

# ── Chunking & Embedding ──────────────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:min(start + CHUNK_SIZE, len(text))].strip()
        if chunk: chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

async def embed_texts(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": texts, "dimensions": 512})
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI embed error: {r.text}")
    data = r.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [d["embedding"] for d in data]

async def embed_query(text: str) -> List[float]:
    return (await embed_texts([text]))[0]

async def upsert_vectors(vectors, host, api_key, namespace):
    h = host if host.startswith("http") else f"https://{host}"
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(vectors), 100):
            r = await client.post(f"{h}/vectors/upsert",
                headers={"Api-Key": api_key, "Content-Type": "application/json"},
                json={"vectors": vectors[i:i+100], "namespace": namespace})
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Upsert error: {r.text}")

# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(); yield

app = FastAPI(title="MCP Pinecone Server", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "llama_parse": bool(LLAMA_API_KEY), "embed_model": EMBED_MODEL}

# ── Admin: Clients ────────────────────────────────────────────────────────────

class ClientCreate(BaseModel):
    name: str; email: Optional[str] = None
    pinecone_api_key: str; pinecone_host: str
    namespace: str = "default"; notes: Optional[str] = None

@app.post("/admin/clients")
def create_client(payload: ClientCreate, _=Depends(get_admin)):
    cid = str(uuid.uuid4())
    key = f"mcp_{secrets.token_urlsafe(32)}"
    conn = get_conn()
    conn.execute("INSERT INTO clients (id,name,email,api_key,pinecone_api_key,pinecone_host,namespace,notes) VALUES (?,?,?,?,?,?,?,?)",
                 (cid, payload.name, payload.email, key, payload.pinecone_api_key, payload.pinecone_host, payload.namespace, payload.notes))
    conn.commit(); conn.close()
    return {"client_id": cid, "name": payload.name, "api_key": key, "namespace": payload.namespace}

@app.get("/admin/clients")
def list_clients(_=Depends(get_admin)):
    conn = get_conn()
    rows = conn.execute("SELECT id,name,email,namespace,active,created_at,notes,substr(api_key,1,12)||'...' as api_key_preview FROM clients ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/admin/clients/{client_id}/rotate-key")
def rotate_key(client_id: str, _=Depends(get_admin)):
    new_key = f"mcp_{secrets.token_urlsafe(32)}"
    conn = get_conn()
    conn.execute("UPDATE clients SET api_key=? WHERE id=?", (new_key, client_id))
    conn.commit(); conn.close()
    return {"api_key": new_key}

@app.get("/admin/clients/{client_id}/usage")
def client_usage(client_id: str, _=Depends(get_admin)):
    conn = get_conn()
    usage = conn.execute("SELECT endpoint,COUNT(*) as calls FROM api_usage WHERE client_id=? GROUP BY endpoint", (client_id,)).fetchall()
    logs = conn.execute("SELECT filename,chunks,status,created_at FROM ingestion_log WHERE client_id=? ORDER BY created_at DESC LIMIT 50", (client_id,)).fetchall()
    conn.close()
    return {"usage": [dict(r) for r in usage], "ingestions": [dict(r) for r in logs]}

# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest/document")
async def ingest_document(file: UploadFile = File(...),
                          namespace_override: Optional[str] = Form(None),
                          extra_metadata: Optional[str] = Form(None),
                          client=Depends(get_client)):
    namespace = namespace_override or client["namespace"]
    text = await extract_text(file)
    if not text.strip(): raise HTTPException(status_code=422, detail="Documento vazio.")
    chunks = chunk_text(text)
    embeddings = await embed_texts(chunks)
    extra = json.loads(extra_metadata) if extra_metadata else {}
    doc_id = hashlib.md5(file.filename.encode()).hexdigest()[:8]
    vectors = [{"id": f"{doc_id}_chunk_{i}", "values": emb,
                "metadata": {"source": file.filename, "chunk_index": i, "total_chunks": len(chunks), "text": c, **extra}}
               for i, (c, emb) in enumerate(zip(chunks, embeddings))]
    await upsert_vectors(vectors, client["pinecone_host"], client["pinecone_api_key"], namespace)
    log_ingestion(client["id"], file.filename, len(chunks), "success")
    log_usage(client["id"], "/ingest/document")
    return {"status": "success", "filename": file.filename, "chunks_ingested": len(chunks), "namespace": namespace, "parser": "llamaparse"}

@app.post("/ingest/text")
async def ingest_text(text: str = Form(...), source_name: str = Form("manual_input"),
                      namespace_override: Optional[str] = Form(None),
                      client=Depends(get_client)):
    namespace = namespace_override or client["namespace"]
    chunks = chunk_text(text)
    embeddings = await embed_texts(chunks)
    doc_id = hashlib.md5(source_name.encode()).hexdigest()[:8]
    vectors = [{"id": f"{doc_id}_chunk_{i}_{uuid.uuid4().hex[:4]}", "values": emb,
                "metadata": {"source": source_name, "chunk_index": i, "text": c, "client_id": client["id"]}}
               for i, (c, emb) in enumerate(zip(chunks, embeddings))]
    await upsert_vectors(vectors, client["pinecone_host"], client["pinecone_api_key"], namespace)
    log_ingestion(client["id"], source_name, len(chunks), "success")
    log_usage(client["id"], "/ingest/text")
    return {"status": "success", "source": source_name, "chunks_ingested": len(chunks), "namespace": namespace}

# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str; top_k: int = 5
    namespace_override: Optional[str] = None
    filter: Optional[dict] = None

async def pinecone_search(query: str, top_k: int, namespace: str, client: dict) -> list:
    host = client["pinecone_host"]
    if not host.startswith("http"): host = f"https://{host}"
    vector = await embed_query(query)
    body = {"vector": vector, "topK": top_k, "includeMetadata": True, "includeValues": False, "namespace": namespace}
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post(f"{host}/query", headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"}, json=body)
    if r.status_code != 200: raise HTTPException(status_code=502, detail=r.text)
    return r.json().get("matches", [])

async def generate_answer(query: str, chunks: list, client_name: str) -> str:
    context = "

---

".join([
        f"[Fonte: {c.get(chr(39)metadata{chr(39)}, {}).get(chr(39)source{chr(39)}, chr(39)desconhecido{chr(39)})}]
{c.get(chr(39)metadata{chr(39)}, {}).get(chr(39)text{chr(39)}, chr(39){chr(39)})}"
        for c in chunks
    ])
    messages = [
        {"role": "system", "content": (
            f"Voce e um assistente especializado no conteudo de {client_name}. "
            "Responda com base APENAS nos trechos fornecidos abaixo. "
            "Se a informacao nao estiver nos trechos, diga que nao encontrou. "
            "Seja direto, claro e estruturado. Use listas quando fizer sentido."
        )},
        {"role": "user", "content": f"Pergunta: {query}

Trechos relevantes:
{context}"}
    ]
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0.3})
    if r.status_code != 200: raise HTTPException(status_code=502, detail=f"GPT error: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

@app.post("/query/search")
async def search(payload: QueryRequest, client=Depends(get_client)):
    namespace = payload.namespace_override or client["namespace"]
    matches = await pinecone_search(payload.query, payload.top_k, namespace, client)
    log_usage(client["id"], "/query/search")
    return {"query": payload.query, "namespace": namespace, "total_matches": len(matches),
            "results": [{"id": m["id"], "score": round(m["score"], 4),
                         "text": m.get("metadata", {}).get("text", ""),
                         "source": m.get("metadata", {}).get("source", ""),
                         "metadata": m.get("metadata", {})} for m in matches]}

@app.post("/query/ask")
async def ask(payload: QueryRequest, client=Depends(get_client)):
    namespace = payload.namespace_override or client["namespace"]
    matches = await pinecone_search(payload.query, payload.top_k, namespace, client)
    if not matches:
        return {"query": payload.query, "answer": "Nao encontrei informacoes sobre isso na base.", "sources": []}
    context = "

---

".join([
        f"[Fonte: {m.get(chr(39)metadata{chr(39)},{}).get(chr(39)source{chr(39)},chr(39){chr(39)})}]
{m.get(chr(39)metadata{chr(39)},{}).get(chr(39)text{chr(39)},chr(39){chr(39)})}"
        for m in matches
    ])
    messages = [
        {"role": "system", "content": f"Voce e assistente especializado no conteudo de {client[chr(39)name{chr(39)}]}. Responda com base APENAS nos trechos. Seja claro e estruturado."},
        {"role": "user", "content": f"Pergunta: {payload.query}

Trechos:
{context}"}
    ]
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0.3})
    if r.status_code != 200: raise HTTPException(status_code=502, detail=f"GPT error: {r.text}")
    answer = r.json()["choices"][0]["message"]["content"]
    sources = list(set([m.get("metadata", {}).get("source", "") for m in matches]))
    log_usage(client["id"], "/query/ask")
    return {"query": payload.query, "answer": answer, "sources": sources, "chunks_used": len(matches)}

@app.get("/query/stats")
async def stats(client=Depends(get_client)):
    host = client["pinecone_host"]
    if not host.startswith("http"): host = f"https://{host}"
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.post(f"{host}/describe_index_stats",
            headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"}, json={})
    return r.json()

# ── MCP ───────────────────────────────────────────────────────────────────────

MCP_TOOLS = [
    {"name": "search_knowledge_base",
     "description": "Busca semântica na base de conhecimento do cliente.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "top_k": {"type": "integer", "default": 5},
         "namespace": {"type": "string"}}, "required": ["query"]}},
    {"name": "ingest_text",
     "description": "Ingere texto na base de conhecimento.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string"}, "source_name": {"type": "string"},
         "namespace": {"type": "string"}}, "required": ["text"]}},
    {"name": "get_index_stats",
     "description": "Retorna estatísticas do índice Pinecone.",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
]

@app.get("/mcp")
async def mcp_discover(client=Depends(get_client)):
    return {"schema_version": "v1", "name": f"kb-{client['name'].lower().replace(' ','-')}",
            "description": f"Base de conhecimento de {client['name']}", "tools": MCP_TOOLS}

@app.post("/mcp/call")
async def mcp_call(request: Request, client=Depends(get_client)):
    body = await request.json()
    name = body.get("name"); inp = body.get("input", {})
    namespace = inp.get("namespace") or client["namespace"]
    log_usage(client["id"], f"/mcp/call:{name}")
    if name == "search_knowledge_base":
        result = await search(QueryRequest(query=inp["query"], top_k=inp.get("top_k", 5),
                                           namespace_override=namespace), client)
        return {"type": "tool_result", "content": result}
    elif name == "ingest_text":
        chunks = chunk_text(inp["text"])
        embeddings = await embed_texts(chunks)
        source = inp.get("source_name", "mcp_input")
        doc_id = hashlib.md5(source.encode()).hexdigest()[:8]
        vectors = [{"id": f"{doc_id}_chunk_{i}_{uuid.uuid4().hex[:4]}", "values": emb,
                    "metadata": {"source": source, "chunk_index": i, "text": c}}
                   for i, (c, emb) in enumerate(zip(chunks, embeddings))]
        await upsert_vectors(vectors, client["pinecone_host"], client["pinecone_api_key"], namespace)
        return {"type": "tool_result", "content": {"status": "success", "chunks_ingested": len(chunks)}}
    elif name == "get_index_stats":
        result = await stats(client)
        return {"type": "tool_result", "content": result}
    raise HTTPException(status_code=404, detail=f"Tool '{name}' não encontrada")
