"""
Query semântica no Pinecone
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import httpx

from core.auth import get_client
from core.database import log_usage

router = APIRouter()

EMBED_MODEL = "multilingual-e5-large"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    namespace_override: Optional[str] = None
    filter: Optional[dict] = None
    include_metadata: bool = True

async def embed_query(text: str, pinecone_api_key: str) -> List[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.pinecone.io/embed",
            headers={"Api-Key": pinecone_api_key, "Content-Type": "application/json"},
            json={
                "model": EMBED_MODEL,
                "inputs": [{"text": text}],
                "parameters": {"input_type": "query", "truncate": "END"},
            },
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Pinecone embed error: {resp.text}")
    return resp.json()["data"][0]["values"]

@router.post("/search")
async def semantic_search(payload: QueryRequest, client: dict = Depends(get_client)):
    """Busca semântica no índice do cliente"""
    namespace = payload.namespace_override or client["namespace"]
    host = client["pinecone_host"]
    if not host.startswith("http"):
        host = f"https://{host}"

    vector = await embed_query(payload.query, client["pinecone_api_key"])

    body = {
        "vector": vector,
        "topK": payload.top_k,
        "includeMetadata": payload.include_metadata,
        "includeValues": False,
        "namespace": namespace,
    }
    if payload.filter:
        body["filter"] = payload.filter

    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            f"{host}/query",
            headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"},
            json=body,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Pinecone query error: {resp.text}")

    data = resp.json()
    log_usage(client["id"], "/query/search")

    matches = data.get("matches", [])
    return {
        "query": payload.query,
        "namespace": namespace,
        "total_matches": len(matches),
        "results": [
            {
                "id": m["id"],
                "score": round(m["score"], 4),
                "text": m.get("metadata", {}).get("text", ""),
                "source": m.get("metadata", {}).get("source", ""),
                "metadata": m.get("metadata", {}),
            }
            for m in matches
        ],
    }

@router.get("/stats")
async def index_stats(client: dict = Depends(get_client)):
    """Estatísticas do índice do cliente"""
    host = client["pinecone_host"]
    if not host.startswith("http"):
        host = f"https://{host}"
    
    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.post(
            f"{host}/describe_index_stats",
            headers={"Api-Key": client["pinecone_api_key"], "Content-Type": "application/json"},
            json={},
        )
    
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=resp.text)
    
    return resp.json()
