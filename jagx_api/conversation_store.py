"""SQLite conversation history isolated by server-derived API-key namespace.

For true account-level isolation, replace API-key identity with a verified JWT subject.
"""
from __future__ import annotations
import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from jagx_api.config import settings

def _db_path() -> Path:
    return Path(getattr(settings, "conversation_db", "jagx_conversations.sqlite3"))

def _owner(api_key: str) -> str:
    return hashlib.sha256(("jagx-conversation-v1:" + api_key).encode()).hexdigest()

def _connect():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
        owner TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
        provider TEXT, created_at TEXT NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id))""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_conv_owner_updated ON conversations(owner, updated_at)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_msg_owner_conv ON messages(owner, conversation_id, id)")
    return con

def create(api_key: str, title: str = "New conversation") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    cid = str(uuid.uuid4())
    with _connect() as con:
        con.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                    (cid, _owner(api_key), (title or "New conversation")[:160], now, now))
    return {"id": cid, "title": (title or "New conversation")[:160], "created_at": now, "updated_at": now, "messages": []}

def list_for(api_key: str, limit: int = 50) -> list:
    with _connect() as con:
        rows = con.execute("SELECT id,title,created_at,updated_at FROM conversations WHERE owner=? ORDER BY updated_at DESC LIMIT ?",
                           (_owner(api_key), max(1,min(int(limit),100)))).fetchall()
    return [dict(r) for r in rows]

def get(api_key: str, conversation_id: str, limit: int = 100) -> dict | None:
    owner = _owner(api_key)
    with _connect() as con:
        row = con.execute("SELECT id,title,created_at,updated_at FROM conversations WHERE id=? AND owner=?",
                          (conversation_id,owner)).fetchone()
        if not row:
            return None
        msgs = con.execute("SELECT role,content,provider,created_at FROM messages WHERE conversation_id=? AND owner=? ORDER BY id DESC LIMIT ?",
                           (conversation_id,owner,max(1,min(int(limit),300)))).fetchall()
    result = dict(row)
    result["messages"] = [dict(m) for m in reversed(msgs)]
    return result

def append(api_key: str, conversation_id: str, role: str, content: str, provider: str | None = None) -> bool:
    if role not in {"user","assistant"}:
        raise ValueError("Invalid message role")
    owner = _owner(api_key)
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        found = con.execute("SELECT 1 FROM conversations WHERE id=? AND owner=?",(conversation_id,owner)).fetchone()
        if not found:
            return False
        con.execute("INSERT INTO messages(conversation_id,owner,role,content,provider,created_at) VALUES(?,?,?,?,?,?)",
                    (conversation_id,owner,role,(content or "")[:32000],provider,now))
        con.execute("UPDATE conversations SET updated_at=? WHERE id=? AND owner=?",(now,conversation_id,owner))
    return True

def delete(api_key: str, conversation_id: str) -> bool:
    owner = _owner(api_key)
    with _connect() as con:
        found = con.execute("SELECT 1 FROM conversations WHERE id=? AND owner=?",(conversation_id,owner)).fetchone()
        if not found:
            return False
        con.execute("DELETE FROM messages WHERE conversation_id=? AND owner=?",(conversation_id,owner))
        con.execute("DELETE FROM conversations WHERE id=? AND owner=?",(conversation_id,owner))
    return True
