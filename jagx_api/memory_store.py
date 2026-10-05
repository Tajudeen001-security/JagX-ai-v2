"""Per-user memory facts — only injected when relevant."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jagx_api.config import settings

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path() -> Path:
    return Path(settings.memory_file)


def _load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    _path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def remember(user_id: str, text: str, tags: str = "") -> dict[str, Any]:
    user_id = (user_id or "guest").strip()[:120]
    text = (text or "").strip()
    if len(text) < 2:
        return {"ok": False, "error": "text required"}
    item = {
        "id": str(uuid.uuid4()),
        "text": text[:800],
        "tags": (tags or "")[:120],
        "ts": _now(),
    }
    with _lock:
        data = _load()
        rows = data.get(user_id) or []
        rows.insert(0, item)
        data[user_id] = rows[:200]
        _save(data)
    return {"ok": True, "id": item["id"]}


def recall(user_id: str, query: str, limit: int = 5) -> list:
    user_id = (user_id or "guest").strip()[:120]
    words = set(re.findall(r"[a-z0-9]{3,}", (query or "").lower()))
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


def forget(user_id: str, memory_id: str = "") -> dict[str, Any]:
    user_id = (user_id or "guest").strip()[:120]
    with _lock:
        data = _load()
        rows = data.get(user_id) or []
        if memory_id:
            new_rows = [r for r in rows if r.get("id") != memory_id]
            removed = len(rows) - len(new_rows)
        else:
            new_rows = []
            removed = len(rows)
        data[user_id] = new_rows
        _save(data)
    return {"ok": True, "removed": removed}
