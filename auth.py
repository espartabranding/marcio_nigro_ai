"""
SQLite database para gerenciar clientes e API keys
"""

import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "mcp_server.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS clients (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            email       TEXT,
            api_key     TEXT UNIQUE NOT NULL,
            pinecone_api_key  TEXT NOT NULL,
            pinecone_host     TEXT NOT NULL,
            namespace         TEXT NOT NULL DEFAULT 'default',
            active      INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS api_usage (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id   TEXT NOT NULL,
            endpoint    TEXT NOT NULL,
            tokens_used INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (client_id) REFERENCES clients(id)
        );

        CREATE TABLE IF NOT EXISTS ingestion_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id   TEXT NOT NULL,
            filename    TEXT NOT NULL,
            chunks      INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL DEFAULT 'pending',
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (client_id) REFERENCES clients(id)
        );
    """)
    conn.commit()
    conn.close()

def log_usage(client_id: str, endpoint: str, tokens: int = 0):
    conn = get_conn()
    conn.execute(
        "INSERT INTO api_usage (client_id, endpoint, tokens_used) VALUES (?, ?, ?)",
        (client_id, endpoint, tokens)
    )
    conn.commit()
    conn.close()

def log_ingestion(client_id: str, filename: str, chunks: int, status: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO ingestion_log (client_id, filename, chunks, status) VALUES (?, ?, ?, ?)",
        (client_id, filename, chunks, status)
    )
    conn.commit()
    conn.close()
