"""Per-user memory facts — only injected when relevant.

When JAGX_DATA_ENCRYPTION_KEY is configured, the entire memory JSON file is
Fernet-encrypted at rest. Legacy plaintext JSON remains readable and is
converted to encrypted storage on the next successful write.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from jagx_api.config import settings

_lock = threading.Lock()
_PREFIX = "enc:v1:"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path() -> Path:
    return Path(settings.memory_file)


def _cipher():
    key = getattr(settings, "data_encryption_key", "")
    if not key:
        return None
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError("JAGX_DATA_ENCRYPTION_KEY must be a valid Fernet key") from exc


def _load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    raw = p.read_text(encoding="utf-8")
    if raw.startswith(_PREFIX):
        cipher = _cipher()
        if cipher is None:
            raise RuntimeError("Encrypted memory exists but JAGX_DATA_ENCRYPTION_KEY is not configured")
        try:
            raw = cipher.decrypt(raw[len(_PREFIX):].encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeEncodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Could not decrypt memory data; verify JAGX_DATA_ENCRYPTION_KEY") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Do not silently discard encrypted or malformed user memory.
        raise RuntimeError("Memory file is not valid JSON; preserve it and investigate before writing")
    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    cipher = _cipher()
    if cipher is not None:
        serialized = _PREFIX + cipher.encrypt(serialized.encode("utf-8")).decode("ascii")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(serialized, encoding="utf-8")
    try:
        temp.chmod(0o600)
    except OSError:
        pass
    temp.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def remember(user_id: str, text: str, tags: str = "") -> dict[str, Any]:
    user_id = (user_id or "guest").strip()[:120]
    text = (text or "").strip()
    if len(text) < 2:
        return {"ok": False, "error": "text required"}
    item = {"id": str(uuid.uuid4()), "text": text[:800], "tags": (tags or "")[:120], "ts": _now()}
    with _lock:
        data = _load()
        rows = data.get(user_id) or []
        rows.insert(0, item)
        data[user_id] = rows[:200]
        _save(data)
    return {"ok": True, "id": item["id"]}


def recall(user_id: str, query: str, limit: int = 5) -> list:
    user_id = (user_id or "guest").strip()[:120]
    words = set(re.findall(r"[^\\W_]{3,}", (query or "").lower()))
    if not words:
        return []
    with _lock:
        rows = list(_load().get(user_id) or [])
    scored = []
    for row in rows:
        blob = (row.get("text", "") + " " + row.get("tags", "")).lower()
        hit = sum(1 for word in words if word in blob)
        if hit:
            scored.append((hit, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[: max(1, min(limit, 8))]]


def forget(user_id: str, memory_id: str = "") -> dict[str, Any]:
    user_id = (user_id or "guest").strip()[:120]
    with _lock:
        data = _load()
        rows = data.get(user_id) or []
        if memory_id:
            new_rows = [row for row in rows if row.get("id") != memory_id]
            removed = len(rows) - len(new_rows)
        else:
            new_rows = []
            removed = len(rows)
        data[user_id] = new_rows
        _save(data)
    return {"ok": True, "removed": removed}
