"""
Autenticação por API Key
"""

from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader
from core.database import get_conn

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_client(x_api_key: str = Security(api_key_header)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header obrigatório")
    
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM clients WHERE api_key = ? AND active = 1",
        (x_api_key,)
    ).fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=403, detail="API Key inválida ou cliente inativo")
    
    return dict(row)

def get_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")):
    """Chave de admin para gerenciar clientes — lida do env"""
    import os
    admin_key = os.getenv("ADMIN_KEY", "admin-change-me-in-production")
    if x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Admin key inválida")
    return True
