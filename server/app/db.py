"""JagX v12 — SQLite persistence layer (WAL).

One module owns the schema and every query. Single-instance by design; swap
this file for Postgres before horizontal scaling.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import settings

_local = threading.local()
_write_lock = threading.Lock()

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT 'New task',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    mode            TEXT NOT NULL DEFAULT 'loop',
    steps_used      INTEGER NOT NULL DEFAULT 0,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    finished_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_user ON runs(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS steps (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL DEFAULT '{}',
    output      TEXT NOT NULL DEFAULT '',
    ok          INTEGER NOT NULL DEFAULT 1,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, idx);

CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    layer        TEXT NOT NULL DEFAULT 'semantic',
    content      TEXT NOT NULL,
    tags         TEXT NOT NULL DEFAULT '',
    vector       TEXT NOT NULL DEFAULT '[]',
    strength     REAL NOT NULL DEFAULT 1.0,
    source_run   TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    last_used_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, layer, created_at DESC);

CREATE TABLE IF NOT EXISTS goals (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    title            TEXT NOT NULL,
    success_criterion TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'active',
    priority         INTEGER NOT NULL DEFAULT 3,
    steps_used       INTEGER NOT NULL DEFAULT 0,
    budget           INTEGER NOT NULL DEFAULT 20,
    next_run_at      REAL NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status, priority);

CREATE TABLE IF NOT EXISTS goal_checkpoints (
    id         TEXT PRIMARY KEY,
    goal_id    TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'ok',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ckpt_goal ON goal_checkpoints(goal_id, idx);

CREATE TABLE IF NOT EXISTS lessons (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    run_id     TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL DEFAULT 'reflection',
    text       TEXT NOT NULL,
    score      REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_user ON lessons(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS prompt_variants (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    task_class TEXT NOT NULL,
    name       TEXT NOT NULL,
    body       TEXT NOT NULL,
    wins       INTEGER NOT NULL DEFAULT 0,
    losses     INTEGER NOT NULL DEFAULT 0,
    score      REAL NOT NULL DEFAULT 0,
    active     INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_variants_user ON prompt_variants(user_id, task_class, active);

CREATE TABLE IF NOT EXISTS evals (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    name       TEXT NOT NULL,
    cases      TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    id         TEXT PRIMARY KEY,
    eval_id    TEXT NOT NULL,
    variant_id TEXT NOT NULL DEFAULT '',
    score      REAL NOT NULL DEFAULT 0,
    detail     TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evalres_eval ON eval_results(eval_id, created_at DESC);

CREATE TABLE IF NOT EXISTS approvals (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    run_id     TEXT NOT NULL DEFAULT '',
    action     TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    decided_at REAL
);
CREATE INDEX IF NOT EXISTS idx_appr_user ON approvals(user_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS settings (
    user_id TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""


def _now() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def db_path() -> Path:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir / "jagx.db"


def connect() -> sqlite3.Connection:
    """One connection per thread (SQLite objects are not thread-safe)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(db_path()), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


def init_db() -> None:
    with _write_lock:
        conn = connect()
        conn.executescript(SCHEMA)
        conn.commit()


def _rows(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    cur = connect().execute(sql, tuple(params))
    return [dict(r) for r in cur.fetchall()]


def _row(sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
    rows = _rows(sql, params)
    return rows[0] if rows else None


def _write(sql: str, params: Iterable[Any] = ()) -> None:
    with _write_lock:
        conn = connect()
        conn.execute(sql, tuple(params))
        conn.commit()


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
def create_user(email: str, password_hash: str) -> dict:
    uid = new_id("usr")
    _write(
        "INSERT INTO users (id, email, password_hash, created_at) VALUES (?,?,?,?)",
        (uid, email.lower().strip(), password_hash, _now()),
    )
    return {"id": uid, "email": email.lower().strip()}


def get_user_by_email(email: str) -> Optional[dict]:
    return _row("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))


def get_user(user_id: str) -> Optional[dict]:
    return _row("SELECT * FROM users WHERE id = ?", (user_id,))


# ---------------------------------------------------------------------------
# conversations & messages
# ---------------------------------------------------------------------------
def create_conversation(user_id: str, title: str = "New task") -> dict:
    cid = new_id("cnv")
    ts = _now()
    _write(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
        (cid, user_id, title[:120], ts, ts),
    )
    return {"id": cid, "user_id": user_id, "title": title[:120], "created_at": ts, "updated_at": ts}


def list_conversations(user_id: str, limit: int = 100) -> list[dict]:
    return _rows(
        "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
        (user_id, limit),
    )


def get_conversation(user_id: str, conversation_id: str) -> Optional[dict]:
    return _row(
        "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )


def rename_conversation(user_id: str, conversation_id: str, title: str) -> None:
    _write(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
        (title[:120], _now(), conversation_id, user_id),
    )


def touch_conversation(conversation_id: str) -> None:
    _write("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id))


def delete_conversation(user_id: str, conversation_id: str) -> None:
    with _write_lock:
        conn = connect()
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
        )
        conn.commit()


def add_message(conversation_id: str, role: str, content: str) -> dict:
    mid = new_id("msg")
    ts = _now()
    _write(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
        (mid, conversation_id, role, content, ts),
    )
    touch_conversation(conversation_id)
    return {"id": mid, "role": role, "content": content, "created_at": ts}


def list_messages(conversation_id: str, limit: int = 200) -> list[dict]:
    rows = _rows(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
        (conversation_id, limit),
    )
    return rows


# ---------------------------------------------------------------------------
# runs & steps
# ---------------------------------------------------------------------------
def create_run(conversation_id: str, user_id: str, mode: str = "loop") -> str:
    rid = new_id("run")
    _write(
        "INSERT INTO runs (id, conversation_id, user_id, status, mode, created_at) VALUES (?,?,?,?,?,?)",
        (rid, conversation_id, user_id, "running", mode, _now()),
    )
    return rid


def finish_run(run_id: str, *, status: str, steps_used: int, provider: str, model: str) -> None:
    _write(
        "UPDATE runs SET status=?, steps_used=?, provider=?, model=?, finished_at=? WHERE id=?",
        (status, steps_used, provider, model, _now(), run_id),
    )


def get_run(user_id: str, run_id: str) -> Optional[dict]:
    return _row("SELECT * FROM runs WHERE id = ? AND user_id = ?", (run_id, user_id))


def list_runs(user_id: str, limit: int = 50) -> list[dict]:
    return _rows(
        "SELECT * FROM runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)
    )


def add_step(
    run_id: str,
    idx: int,
    kind: str,
    *,
    name: str = "",
    payload: Any = None,
    output: str = "",
    ok: bool = True,
    duration_ms: int = 0,
) -> None:
    _write(
        "INSERT INTO steps (id, run_id, idx, kind, name, payload, output, ok, duration_ms, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            new_id("stp"),
            run_id,
            idx,
            kind,
            name,
            json.dumps(payload or {}, ensure_ascii=False)[:20000],
            (output or "")[:20000],
            1 if ok else 0,
            duration_ms,
            _now(),
        ),
    )


def list_steps(run_id: str) -> list[dict]:
    rows = _rows("SELECT * FROM steps WHERE run_id = ? ORDER BY idx ASC", (run_id,))
    for r in rows:
        try:
            r["payload"] = json.loads(r["payload"])
        except (json.JSONDecodeError, TypeError):
            r["payload"] = {}
        r["ok"] = bool(r["ok"])
    return rows


# ---------------------------------------------------------------------------
# memories
# ---------------------------------------------------------------------------
def add_memory(
    user_id: str,
    content: str,
    *,
    layer: str = "semantic",
    tags: str = "",
    vector: Optional[list[float]] = None,
    source_run: str = "",
    strength: float = 1.0,
) -> dict:
    mid = new_id("mem")
    ts = _now()
    _write(
        "INSERT INTO memories (id, user_id, layer, content, tags, vector, strength, source_run,"
        " created_at, last_used_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            mid,
            user_id,
            layer,
            content[:4000],
            tags[:300],
            json.dumps(vector or []),
            strength,
            source_run,
            ts,
            ts,
        ),
    )
    return {"id": mid, "layer": layer, "content": content[:4000], "tags": tags}


def list_memories(user_id: str, *, layer: str = "", limit: int = 200) -> list[dict]:
    if layer:
        rows = _rows(
            "SELECT * FROM memories WHERE user_id = ? AND layer = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, layer, limit),
        )
    else:
        rows = _rows(
            "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
    for r in rows:
        try:
            r["vector"] = json.loads(r["vector"])
        except (json.JSONDecodeError, TypeError):
            r["vector"] = []
    return rows


def all_memories_for_search(user_id: str, limit: int = 2000) -> list[dict]:
    return list_memories(user_id, limit=limit)


def touch_memory(memory_id: str, strength: float) -> None:
    _write(
        "UPDATE memories SET last_used_at = ?, strength = ? WHERE id = ?",
        (_now(), strength, memory_id),
    )


def delete_memory(user_id: str, memory_id: str) -> None:
    _write("DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id))


def decay_memories(user_id: str, factor: float = 0.97, floor: float = 0.15) -> int:
    """Episodic memories fade unless they keep being used."""
    with _write_lock:
        conn = connect()
        cur = conn.execute(
            "UPDATE memories SET strength = MAX(?, strength * ?) WHERE user_id = ? AND layer = 'episodic'",
            (floor, factor, user_id),
        )
        conn.commit()
        return cur.rowcount


def prune_weak_memories(user_id: str, floor: float = 0.2) -> int:
    with _write_lock:
        conn = connect()
        cur = conn.execute(
            "DELETE FROM memories WHERE user_id = ? AND layer = 'episodic' AND strength <= ?",
            (user_id, floor),
        )
        conn.commit()
        return cur.rowcount


# ---------------------------------------------------------------------------
# goals
# ---------------------------------------------------------------------------
def create_goal(
    user_id: str,
    title: str,
    *,
    success_criterion: str = "",
    priority: int = 3,
    budget: int = 20,
) -> dict:
    gid = new_id("goal")
    ts = _now()
    _write(
        "INSERT INTO goals (id, user_id, title, success_criterion, status, priority, steps_used,"
        " budget, next_run_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (gid, user_id, title[:200], success_criterion[:500], "active", priority, 0, budget, 0.0, ts, ts),
    )
    return get_goal(user_id, gid) or {}


def get_goal(user_id: str, goal_id: str) -> Optional[dict]:
    return _row("SELECT * FROM goals WHERE id = ? AND user_id = ?", (goal_id, user_id))


def list_goals(user_id: str, status: str = "") -> list[dict]:
    if status:
        return _rows(
            "SELECT * FROM goals WHERE user_id = ? AND status = ? ORDER BY priority ASC, created_at DESC",
            (user_id, status),
        )
    return _rows(
        "SELECT * FROM goals WHERE user_id = ? ORDER BY priority ASC, created_at DESC", (user_id,)
    )


def update_goal(user_id: str, goal_id: str, **fields: Any) -> None:
    allowed = {"title", "success_criterion", "status", "priority", "steps_used", "budget", "next_run_at"}
    sets, params = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            params.append(v)
    if not sets:
        return
    sets.append("updated_at = ?")
    params.append(_now())
    params.extend([goal_id, user_id])
    _write(f"UPDATE goals SET {', '.join(sets)} WHERE id = ? AND user_id = ?", params)


def delete_goal(user_id: str, goal_id: str) -> None:
    with _write_lock:
        conn = connect()
        conn.execute("DELETE FROM goal_checkpoints WHERE goal_id = ?", (goal_id,))
        conn.execute("DELETE FROM goals WHERE id = ? AND user_id = ?", (goal_id, user_id))
        conn.commit()


def add_checkpoint(goal_id: str, idx: int, note: str, status: str = "ok") -> None:
    _write(
        "INSERT INTO goal_checkpoints (id, goal_id, idx, note, status, created_at) VALUES (?,?,?,?,?,?)",
        (new_id("ckp"), goal_id, idx, note[:2000], status, _now()),
    )


def list_checkpoints(goal_id: str) -> list[dict]:
    return _rows("SELECT * FROM goal_checkpoints WHERE goal_id = ? ORDER BY idx ASC", (goal_id,))


def due_goals(user_id: str, now: Optional[float] = None) -> list[dict]:
    now = now or _now()
    return _rows(
        "SELECT * FROM goals WHERE user_id = ? AND status = 'active' AND next_run_at <= ?"
        " ORDER BY priority ASC, next_run_at ASC",
        (user_id, now),
    )


# ---------------------------------------------------------------------------
# lessons / prompt variants / evals
# ---------------------------------------------------------------------------
def add_lesson(user_id: str, text: str, *, run_id: str = "", kind: str = "reflection", score: float = 0.0) -> dict:
    lid = new_id("les")
    _write(
        "INSERT INTO lessons (id, user_id, run_id, kind, text, score, created_at) VALUES (?,?,?,?,?,?,?)",
        (lid, user_id, run_id, kind, text[:2000], score, _now()),
    )
    return {"id": lid, "text": text[:2000], "kind": kind, "score": score}


def list_lessons(user_id: str, limit: int = 100) -> list[dict]:
    return _rows(
        "SELECT * FROM lessons WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)
    )


def add_variant(user_id: str, task_class: str, name: str, body: str, *, active: bool = True) -> dict:
    vid = new_id("var")
    _write(
        "INSERT INTO prompt_variants (id, user_id, task_class, name, body, wins, losses, score,"
        " active, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (vid, user_id, task_class, name[:80], body, 0, 0, 0.0, 1 if active else 0, _now()),
    )
    return {"id": vid, "task_class": task_class, "name": name, "body": body}


def list_variants(user_id: str, task_class: str = "", active_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM prompt_variants WHERE user_id = ?"
    params: list[Any] = [user_id]
    if task_class:
        sql += " AND task_class = ?"
        params.append(task_class)
    if active_only:
        sql += " AND active = 1"
    sql += " ORDER BY score DESC, created_at DESC"
    return _rows(sql, params)


def record_variant_outcome(variant_id: str, *, won: bool, score: float) -> None:
    if won:
        _write(
            "UPDATE prompt_variants SET wins = wins + 1, score = ? WHERE id = ?", (score, variant_id)
        )
    else:
        _write(
            "UPDATE prompt_variants SET losses = losses + 1, score = ? WHERE id = ?", (score, variant_id)
        )


def set_variant_active(variant_id: str, active: bool) -> None:
    _write("UPDATE prompt_variants SET active = ? WHERE id = ?", (1 if active else 0, variant_id))


def create_eval(user_id: str, name: str, cases: list[dict]) -> dict:
    eid = new_id("eval")
    _write(
        "INSERT INTO evals (id, user_id, name, cases, created_at) VALUES (?,?,?,?,?)",
        (eid, user_id, name[:120], json.dumps(cases, ensure_ascii=False), _now()),
    )
    return {"id": eid, "name": name, "cases": cases}


def list_evals(user_id: str) -> list[dict]:
    rows = _rows("SELECT * FROM evals WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    for r in rows:
        try:
            r["cases"] = json.loads(r["cases"])
        except (json.JSONDecodeError, TypeError):
            r["cases"] = []
    return rows


def get_eval(user_id: str, eval_id: str) -> Optional[dict]:
    row = _row("SELECT * FROM evals WHERE id = ? AND user_id = ?", (eval_id, user_id))
    if row:
        try:
            row["cases"] = json.loads(row["cases"])
        except (json.JSONDecodeError, TypeError):
            row["cases"] = []
    return row


def add_eval_result(eval_id: str, variant_id: str, score: float, detail: Any = None) -> None:
    _write(
        "INSERT INTO eval_results (id, eval_id, variant_id, score, detail, created_at) VALUES (?,?,?,?,?,?)",
        (new_id("evr"), eval_id, variant_id, score, json.dumps(detail or {}, ensure_ascii=False)[:20000], _now()),
    )


def list_eval_results(eval_id: str, limit: int = 50) -> list[dict]:
    rows = _rows(
        "SELECT * FROM eval_results WHERE eval_id = ? ORDER BY created_at DESC LIMIT ?",
        (eval_id, limit),
    )
    for r in rows:
        try:
            r["detail"] = json.loads(r["detail"])
        except (json.JSONDecodeError, TypeError):
            r["detail"] = {}
    return rows


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------
def create_approval(user_id: str, action: str, detail: str, *, run_id: str = "") -> dict:
    aid = new_id("apr")
    _write(
        "INSERT INTO approvals (id, user_id, run_id, action, detail, status, created_at) VALUES (?,?,?,?,?,?,?)",
        (aid, user_id, run_id, action[:200], detail[:2000], "pending", _now()),
    )
    return {"id": aid, "action": action, "detail": detail, "status": "pending"}


def list_approvals(user_id: str, status: str = "") -> list[dict]:
    if status:
        return _rows(
            "SELECT * FROM approvals WHERE user_id = ? AND status = ? ORDER BY created_at DESC",
            (user_id, status),
        )
    return _rows("SELECT * FROM approvals WHERE user_id = ? ORDER BY created_at DESC", (user_id,))


def decide_approval(user_id: str, approval_id: str, status: str) -> None:
    _write(
        "UPDATE approvals SET status = ?, decided_at = ? WHERE id = ? AND user_id = ?",
        (status, _now(), approval_id, user_id),
    )


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS: dict[str, Any] = {
    "model": "",
    "temperature": settings.temperature,
    "max_steps": settings.max_steps,
    "system_prompt": "",
    "auto_memory": True,
    "show_activity": True,
    "critique": True,
    "council_threshold": 0.6,
    "guard_rules": "",
    "theme": "dark",
}


def get_settings(user_id: str) -> dict:
    rows = _rows("SELECT key, value FROM settings WHERE user_id = ?", (user_id,))
    out = dict(DEFAULT_SETTINGS)
    for r in rows:
        try:
            out[r["key"]] = json.loads(r["value"])
        except (json.JSONDecodeError, TypeError):
            out[r["key"]] = r["value"]
    return out


def put_settings(user_id: str, values: dict) -> dict:
    with _write_lock:
        conn = connect()
        for k, v in values.items():
            if k not in DEFAULT_SETTINGS:
                continue
            conn.execute(
                "INSERT INTO settings (user_id, key, value) VALUES (?,?,?)"
                " ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
                (user_id, k, json.dumps(v, ensure_ascii=False)),
            )
        conn.commit()
    return get_settings(user_id)
