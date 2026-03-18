"""
Gerenciamento de clientes e API keys
Protegido por X-Admin-Key
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import secrets
import uuid

from core.database import get_conn
from core.auth import get_admin

router = APIRouter()

class ClientCreate(BaseModel):
    name: str
    email: Optional[str] = None
    pinecone_api_key: str
    pinecone_host: str
    namespace: str = "default"
    notes: Optional[str] = None

class ClientUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_host: Optional[str] = None
    namespace: Optional[str] = None
    active: Optional[bool] = None
    notes: Optional[str] = None

def generate_api_key():
    return f"mcp_{secrets.token_urlsafe(32)}"

@router.post("/clients")
def create_client(payload: ClientCreate, _: bool = Depends(get_admin)):
    """Cria novo cliente e gera API key"""
    client_id = str(uuid.uuid4())
    api_key = generate_api_key()
    
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO clients (id, name, email, api_key, pinecone_api_key, pinecone_host, namespace, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            client_id,
            payload.name,
            payload.email,
            api_key,
            payload.pinecone_api_key,
            payload.pinecone_host,
            payload.namespace,
            payload.notes,
        ))
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()
    
    return {
        "client_id": client_id,
        "name": payload.name,
        "api_key": api_key,
        "namespace": payload.namespace,
        "message": "Cliente criado. Guarde a api_key — ela não será exibida novamente.",
    }

@router.get("/clients")
def list_clients(_: bool = Depends(get_admin)):
    """Lista todos os clientes (sem expor as chaves do Pinecone)"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT id, name, email, namespace, active, created_at, notes,
               substr(api_key, 1, 12) || '...' as api_key_preview
        FROM clients
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@router.get("/clients/{client_id}")
def get_client_detail(client_id: str, _: bool = Depends(get_admin)):
    conn = get_conn()
    row = conn.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Cliente não encontrado")
    d = dict(row)
    d["api_key"] = d["api_key"][:12] + "..."
    return d

@router.patch("/clients/{client_id}")
def update_client(client_id: str, payload: ClientUpdate, _: bool = Depends(get_admin)):
    fields = {k: v for k, v in payload.dict().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="Nenhum campo para atualizar")
    if "active" in fields:
        fields["active"] = 1 if fields["active"] else 0
    
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [client_id]
    
    conn = get_conn()
    conn.execute(f"UPDATE clients SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()
    return {"message": "Cliente atualizado"}

@router.post("/clients/{client_id}/rotate-key")
def rotate_key(client_id: str, _: bool = Depends(get_admin)):
    """Gera nova API key para o cliente"""
    new_key = generate_api_key()
    conn = get_conn()
    conn.execute("UPDATE clients SET api_key = ? WHERE id = ?", (new_key, client_id))
    conn.commit()
    conn.close()
    return {"api_key": new_key, "message": "Chave rotacionada. Guarde — não será exibida novamente."}

@router.get("/clients/{client_id}/usage")
def client_usage(client_id: str, _: bool = Depends(get_admin)):
    conn = get_conn()
    usage = conn.execute("""
        SELECT endpoint, COUNT(*) as calls, SUM(tokens_used) as total_tokens
        FROM api_usage WHERE client_id = ?
        GROUP BY endpoint
    """, (client_id,)).fetchall()
    ingestions = conn.execute("""
        SELECT filename, chunks, status, created_at
        FROM ingestion_log WHERE client_id = ?
        ORDER BY created_at DESC LIMIT 50
    """, (client_id,)).fetchall()
    conn.close()
    return {
        "usage_by_endpoint": [dict(r) for r in usage],
        "ingestion_history": [dict(r) for r in ingestions],
    }

@router.delete("/clients/{client_id}")
def deactivate_client(client_id: str, _: bool = Depends(get_admin)):
    conn = get_conn()
    conn.execute("UPDATE clients SET active = 0 WHERE id = ?", (client_id,))
    conn.commit()
    conn.close()
    return {"message": "Cliente desativado"}
