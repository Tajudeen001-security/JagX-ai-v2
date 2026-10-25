"""Durable SQLite job queue for work that must outlive a client connection.

Jobs survive process restarts when the database lives on persistent storage.
V10: priority queue (paid tiers first), worker stats.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Optional

from jagx_api.config import settings

_lock = threading.RLock()

TIER_PRIORITY = {
    "master": 100,
    "admin": 100,
    "premium_plus": 80,
    "premium": 50,
    "free": 10,
}


def _connect():
    db = sqlite3.connect(settings.jobs_db, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=15000")
    db.execute(
        """CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL,
        payload TEXT NOT NULL, status TEXT NOT NULL, result TEXT,
        error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        started_at REAL, finished_at REAL, checkpoint TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        next_attempt_at REAL NOT NULL DEFAULT 0,
        priority INTEGER NOT NULL DEFAULT 10
    )"""
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner_created ON jobs(owner,created_at DESC)")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
    if "checkpoint" not in columns:
        db.execute("ALTER TABLE jobs ADD COLUMN checkpoint TEXT")
    if "cancel_requested" not in columns:
        db.execute("ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
    if "next_attempt_at" not in columns:
        db.execute("ALTER TABLE jobs ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0")
    if "priority" not in columns:
        db.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 10")
    db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status,created_at)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(status, priority DESC, created_at)")
    return db


def init():
    with _lock, _connect() as db:
        db.execute(
            "UPDATE jobs SET status='queued', updated_at=? WHERE status='running' AND cancel_requested=0",
            (time.time(),),
        )
        db.execute(
            "UPDATE jobs SET status='cancelled', finished_at=?, updated_at=? WHERE status='running' AND cancel_requested=1",
            (time.time(), time.time()),
        )
        db.commit()


def enqueue(owner: str, kind: str, payload: dict, *, priority: int = 10) -> dict:
    now = time.time()
    jid = str(uuid.uuid4())
    priority = int(priority)
    with _lock, _connect() as db:
        db.execute(
            "INSERT INTO jobs(id,owner,kind,payload,status,created_at,updated_at,priority) VALUES(?,?,?,?,?,?,?,?)",
            (jid, owner, kind, json.dumps(payload, ensure_ascii=False), "queued", now, now, priority),
        )
        db.commit()
    return get(owner, jid)


def claim_next() -> Optional[dict]:
    now = time.time()
    with _lock, _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM jobs WHERE status='queued' AND next_attempt_at<=? "
            "ORDER BY priority DESC, created_at ASC LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            db.commit()
            return None
        changed = db.execute(
            "UPDATE jobs SET status='running',attempts=attempts+1,started_at=?,updated_at=? "
            "WHERE id=? AND status='queued'",
            (now, now, row["id"]),
        ).rowcount
        db.commit()
        if not changed:
            return None
        return _decode(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())


def finish(job_id: str, result: dict):
    now = time.time()
    with _lock, _connect() as db:
        db.execute(
            "UPDATE jobs SET status='succeeded',result=?,error=NULL,updated_at=?,finished_at=? "
            "WHERE id=? AND status='running'",
            (json.dumps(result, ensure_ascii=False), now, now, job_id),
        )
        db.commit()


def fail(job_id: str, error: str, retry: bool = False):
    now = time.time()
    with _lock, _connect() as db:
        row = db.execute("SELECT attempts FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return
        status = "queued" if retry and row["attempts"] < 3 else "failed"
        delay = min(60, 2 ** max(1, int(row["attempts"]))) if status == "queued" else 0
        db.execute(
            "UPDATE jobs SET status=?,error=?,updated_at=?,finished_at=?,next_attempt_at=? WHERE id=?",
            (status, error[:1200], now, None if status == "queued" else now, now + delay, job_id),
        )
        db.commit()


def cancel(owner: str, job_id: str) -> Optional[dict]:
    now = time.time()
    with _lock, _connect() as db:
        row = db.execute("SELECT status FROM jobs WHERE id=? AND owner=?", (job_id, owner)).fetchone()
        if not row:
            return None
        if row["status"] in ("queued", "running"):
            db.execute(
                "UPDATE jobs SET status='cancelled',updated_at=?,finished_at=? "
                "WHERE id=? AND status IN ('queued','running')",
                (now, now, job_id),
            )
            db.commit()
    return get(owner, job_id)


def _decode(row):
    if row is None:
        return None
    d = {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "payload": json.loads(row["payload"]),
        "result": json.loads(row["result"]) if row["result"] else None,
        "error": row["error"],
        "attempts": row["attempts"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "checkpoint": json.loads(row["checkpoint"]) if row["checkpoint"] else None,
        "cancel_requested": bool(row["cancel_requested"]),
        "next_attempt_at": row["next_attempt_at"],
    }
    try:
        d["priority"] = int(row["priority"]) if row["priority"] is not None else 10
    except (KeyError, IndexError, TypeError):
        d["priority"] = 10
    return d


def get(owner: str, job_id: str) -> Optional[dict]:
    with _lock, _connect() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner)).fetchone()
        return _decode(row)


def list_for(owner: str, limit: int = 20) -> list:
    with _lock, _connect() as db:
        rows = db.execute(
            "SELECT * FROM jobs WHERE owner=? ORDER BY created_at DESC LIMIT ?",
            (owner, max(1, min(int(limit), 100))),
        ).fetchall()
        return [_decode(row) for row in rows]


def checkpoint(job_id: str, value: dict) -> bool:
    now = time.time()
    with _lock, _connect() as db:
        row = db.execute("SELECT status,cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or row["status"] != "running" or row["cancel_requested"]:
            return False
        db.execute(
            "UPDATE jobs SET checkpoint=?,updated_at=? WHERE id=?",
            (json.dumps(value, ensure_ascii=False), now, job_id),
        )
        db.commit()
        return True


def cancellation_requested(job_id: str) -> bool:
    with _lock, _connect() as db:
        row = db.execute("SELECT status,cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return not row or row["status"] != "running" or bool(row["cancel_requested"])


def request_cancel(owner: str, job_id: str) -> Optional[dict]:
    now = time.time()
    with _lock, _connect() as db:
        row = db.execute("SELECT status FROM jobs WHERE id=? AND owner=?", (job_id, owner)).fetchone()
        if not row:
            return None
        if row["status"] == "queued":
            db.execute(
                "UPDATE jobs SET status='cancelled',updated_at=?,finished_at=? WHERE id=?",
                (now, now, job_id),
            )
        elif row["status"] == "running":
            db.execute(
                "UPDATE jobs SET status='cancelled',cancel_requested=1,updated_at=?,finished_at=? WHERE id=?",
                (now, now, job_id),
            )
        db.commit()
    return get(owner, job_id)


def worker_stats() -> dict:
    with _lock, _connect() as db:
        rows = db.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        by_status = {r["status"]: r["n"] for r in rows}
        queued = by_status.get("queued", 0)
        running = by_status.get("running", 0)
        wait_row = db.execute(
            "SELECT AVG(? - created_at) AS avg_wait FROM jobs WHERE status='queued'",
            (time.time(),),
        ).fetchone()
        avg_wait = float(wait_row["avg_wait"] or 0) if wait_row else 0.0
        high_pri = db.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='queued' AND priority>=50"
        ).fetchone()["n"]
    return {
        "queued": queued,
        "running": running,
        "succeeded": by_status.get("succeeded", 0),
        "failed": by_status.get("failed", 0),
        "cancelled": by_status.get("cancelled", 0),
        "avg_queue_wait_seconds": round(avg_wait, 1),
        "high_priority_queued": high_pri,
        "tier_priority_map": TIER_PRIORITY,
    }


def priority_for_tier(tier: str) -> int:
    return TIER_PRIORITY.get((tier or "free").lower(), 10)
