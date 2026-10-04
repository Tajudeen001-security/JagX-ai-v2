"""
JagX memory — facts stored per user, returned only when the question needs them.
Never dumped into every reply.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger("jagx-ai")
FILE = os.environ.get("JAGX_MEMORY_FILE", "jagx_memory.json")
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    if not os.path.exists(FILE):
        return {}
    try:
        with open(FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def remember(user_id: str, text: str, tags: str = "") -> dict:
    user_id = (user_id or "guest").strip()[:120]
    text = (text or "").strip()
    if len(text) < 2:
        return {"ok": False, "error": "text required"}
    item = {
        "id": str(uuid.uuid4()),
        "text": text[:800],
        "tags": tags[:120],
        "ts": _now(),
    }
    with _lock:
        data = _load()
        rows = data.get(user_id) or []
        rows.insert(0, item)
        data[user_id] = rows[:200]
        _save(data)
    return {"ok": True, "id": item["id"]}


def recall(user_id: str, query: str, limit: int = 5) -> List[dict]:
    """Return only memories that share words with the query."""
    user_id = (user_id or "guest").strip()[:120]
    words = set(re.findall(r"[a-z0-9]{4,}", (query or "").lower()))
    if not words:
        return []
    with _lock:
        rows = list(_load().get(user_id) or [])
    scored = []
    for row in rows:
        blob = (row.get("text", "") + " " + row.get("tags", "")).lower()
        hit = sum(1 for w in words if w in blob)
        if hit:
            scored.append((hit, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[: max(1, min(limit, 8))]]


def forget(user_id: str, memory_id: str = "") -> dict:
    user_id = (user_id or "guest").strip()[:120]
    with _lock:
        data = _load()
        rows = data.get(user_id) or []
        if memory_id:
            rows = [r for r in rows if r.get("id") != memory_id]
        else:
            rows = []
        data[user_id] = rows
        _save(data)
    return {"ok": True, "left": len(rows)}


def register_memory_routes(app=None) -> None:
    app = app or globals().get("app")
    if app is None:
        return
    from fastapi import Body

    @app.post("/memory")
    def mem_put(body: dict = Body(...)):
        return remember(str(body.get("user_id") or "guest"), str(body.get("text") or ""), str(body.get("tags") or ""))

    @app.post("/memory/recall")
    def mem_recall(body: dict = Body(...)):
        rows = recall(str(body.get("user_id") or "guest"), str(body.get("query") or ""))
        return {"memories": rows}

    @app.post("/memory/forget")
    def mem_forget(body: dict = Body(...)):
        return forget(str(body.get("user_id") or "guest"), str(body.get("id") or ""))

    logger.info("JagX memory routes registered")
