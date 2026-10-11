"""SQLite-backed conversation history scoped to an authenticated API key."""
from __future__ import annotations
import hashlib, os, sqlite3, time, uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = os.environ.get("JAGX_CONVERSATIONS_DB", "jagx_conversations.sqlite3")
def _owner(api_key: str) -> str:
    return hashlib.sha256(("jagx-conversation-v1:" + api_key).encode()).hexdigest()
@contextmanager
def _db():
    path=Path(DB_PATH)
    if str(path.parent) not in ("", "."): path.parent.mkdir(parents=True, exist_ok=True)
    conn=sqlite3.connect(str(path), timeout=10); conn.row_factory=sqlite3.Row
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, owner_hash TEXT NOT NULL, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated ON conversations(owner_hash, updated_at DESC)")
        conn.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, owner_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL, provider TEXT, created_at REAL NOT NULL)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_owner_conversation ON messages(owner_hash, conversation_id, id)")
        conn.commit(); yield conn
    finally: conn.close()
def create(api_key: str, title: str="New conversation") -> dict:
    owner=_owner(api_key); cid=str(uuid.uuid4()); now=time.time(); title=(title or "New conversation")[:160]
    with _db() as db: db.execute("INSERT INTO conversations VALUES(?,?,?,?,?)",(cid,owner,title,now,now)); db.commit()
    return {"id":cid,"title":title,"created_at":now,"updated_at":now}
def list_for(api_key: str, limit: int=50) -> list[dict]:
    with _db() as db: rows=db.execute("SELECT id,title,created_at,updated_at FROM conversations WHERE owner_hash=? ORDER BY updated_at DESC LIMIT ?",(_owner(api_key),max(1,min(limit,100)))).fetchall()
    return [dict(r) for r in rows]
def get(api_key: str, conversation_id: str, limit: int=100) -> dict|None:
    owner=_owner(api_key)
    with _db() as db:
        conv=db.execute("SELECT id,title,created_at,updated_at FROM conversations WHERE id=? AND owner_hash=?",(conversation_id,owner)).fetchone()
        if not conv: return None
        rows=db.execute("SELECT role,content,provider,created_at FROM messages WHERE conversation_id=? AND owner_hash=? ORDER BY id DESC LIMIT ?",(conversation_id,owner,max(1,min(limit,300)))).fetchall()
    out=dict(conv); out["messages"]=[dict(r) for r in reversed(rows)]; return out
def append(api_key: str, conversation_id: str, role: str, content: str, provider: str|None=None) -> bool:
    if role not in ("user","assistant") or not content: return False
    owner=_owner(api_key); now=time.time()
    with _db() as db:
        if not db.execute("SELECT 1 FROM conversations WHERE id=? AND owner_hash=?",(conversation_id,owner)).fetchone(): return False
        db.execute("INSERT INTO messages(conversation_id,owner_hash,role,content,provider,created_at) VALUES(?,?,?,?,?,?)",(conversation_id,owner,role,content[:32000],(provider or "")[:80],now))
        db.execute("UPDATE conversations SET updated_at=? WHERE id=? AND owner_hash=?",(now,conversation_id,owner)); db.commit()
    return True
def delete(api_key: str, conversation_id: str) -> bool:
    owner=_owner(api_key)
    with _db() as db:
        db.execute("DELETE FROM messages WHERE conversation_id=? AND owner_hash=?",(conversation_id,owner))
        cur=db.execute("DELETE FROM conversations WHERE id=? AND owner_hash=?",(conversation_id,owner)); db.commit()
        return cur.rowcount>0
