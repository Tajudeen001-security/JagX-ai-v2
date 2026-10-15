"""Durable SQLite job queue for work that must outlive a client connection.

Jobs survive process restarts when the database lives on persistent storage.
This module intentionally stores prompts/results only; it does not execute
arbitrary shell commands or write to GitHub on its own.
"""
from __future__ import annotations
import json, sqlite3, threading, time, uuid
from pathlib import Path
from typing import Optional
from jagx_api.config import settings

_lock = threading.RLock()

def _connect():
    db=sqlite3.connect(settings.jobs_db, timeout=15)
    db.row_factory=sqlite3.Row
    db.execute("PRAGMA busy_timeout=15000")
    db.execute("""CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL,
        payload TEXT NOT NULL, status TEXT NOT NULL, result TEXT,
        error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        started_at REAL, finished_at REAL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner_created ON jobs(owner,created_at DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status,created_at)")
    return db

def init():
    with _lock, _connect() as db:
        # A process may have died while a job was running. Re-queue it for recovery.
        db.execute("UPDATE jobs SET status='queued', updated_at=? WHERE status='running'",(time.time(),))
        db.commit()

def enqueue(owner: str, kind: str, payload: dict) -> dict:
    now=time.time(); jid=str(uuid.uuid4())
    with _lock, _connect() as db:
        db.execute("INSERT INTO jobs(id,owner,kind,payload,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                   (jid,owner,kind,json.dumps(payload,ensure_ascii=False),"queued",now,now))
        db.commit()
    return get(owner,jid)

def claim_next() -> Optional[dict]:
    now=time.time()
    with _lock, _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            db.commit(); return None
        changed=db.execute("UPDATE jobs SET status='running',attempts=attempts+1,started_at=?,updated_at=? WHERE id=? AND status='queued'",
                           (now,now,row["id"])).rowcount
        db.commit()
        if not changed: return None
        return _decode(db.execute("SELECT * FROM jobs WHERE id=?",(row["id"],)).fetchone())

def finish(job_id: str, result: dict):
    now=time.time()
    with _lock, _connect() as db:
        db.execute("UPDATE jobs SET status='succeeded',result=?,error=NULL,updated_at=?,finished_at=? WHERE id=? AND status='running'",
                   (json.dumps(result,ensure_ascii=False),now,now,job_id)); db.commit()

def fail(job_id: str, error: str, retry: bool=False):
    now=time.time()
    with _lock, _connect() as db:
        row=db.execute("SELECT attempts FROM jobs WHERE id=?",(job_id,)).fetchone()
        if not row: return
        status="queued" if retry and row["attempts"] < 3 else "failed"
        db.execute("UPDATE jobs SET status=?,error=?,updated_at=?,finished_at=? WHERE id=?",
                   (status,error[:1200],now,None if status=="queued" else now,job_id)); db.commit()

def cancel(owner: str, job_id: str) -> Optional[dict]:
    now=time.time()
    with _lock, _connect() as db:
        row=db.execute("SELECT status FROM jobs WHERE id=? AND owner=?",(job_id,owner)).fetchone()
        if not row: return None
        if row["status"] in ("queued","running"):
            db.execute("UPDATE jobs SET status='cancelled',updated_at=?,finished_at=? WHERE id=? AND status IN ('queued','running')",
                       (now,now,job_id)); db.commit()
    return get(owner,job_id)

def _decode(row):
    if row is None: return None
    return {"id":row["id"],"kind":row["kind"],"status":row["status"],
            "payload":json.loads(row["payload"]),"result":json.loads(row["result"]) if row["result"] else None,
            "error":row["error"],"attempts":row["attempts"],"created_at":row["created_at"],
            "updated_at":row["updated_at"],"started_at":row["started_at"],"finished_at":row["finished_at"]}

def get(owner: str, job_id: str) -> Optional[dict]:
    with _lock, _connect() as db:
        row=db.execute("SELECT * FROM jobs WHERE id=? AND owner=?",(job_id,owner)).fetchone()
        return _decode(row)

def list_for(owner: str, limit: int=20) -> list:
    with _lock, _connect() as db:
        rows=db.execute("SELECT * FROM jobs WHERE owner=? ORDER BY created_at DESC LIMIT ?",(owner,max(1,min(int(limit),100)))).fetchall()
        return [_decode(row) for row in rows]
