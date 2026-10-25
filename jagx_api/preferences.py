"""
JagX AI V9.6 — User style & preference memory (consented).
Lightweight profile injected into multi-agent / chat context.
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from jagx_api.config import settings

_lock = threading.Lock()

_ALLOWED = {
    "tone": {"calm", "direct", "friendly", "formal", "technical"},
    "verbosity": {"short", "normal", "detailed"},
    "language": None,
    "code_style": {"clean", "commented", "minimal", "production"},
    "reply_language": None,
}


def _path() -> Path:
    base = Path(getattr(settings, "memory_file", "jagx_memory.json")).parent
    return base / "jagx_preferences.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        temp.chmod(0o600)
    except OSError:
        pass
    temp.replace(path)


def get_preferences(owner_id: str) -> Dict[str, Any]:
    owner_id = (owner_id or "guest").strip()[:120]
    with _lock:
        row = _load().get(owner_id) or {}
    prefs = row.get("prefs") if isinstance(row.get("prefs"), dict) else {}
    return {
        "ok": True,
        "owner": owner_id,
        "prefs": prefs,
        "updated_at": row.get("updated_at"),
        "consent": bool(row.get("consent")),
    }


def set_preferences(
    owner_id: str,
    prefs: dict,
    *,
    consent: bool = False,
) -> Dict[str, Any]:
    owner_id = (owner_id or "guest").strip()[:120]
    if not consent:
        return {"ok": False, "error": "consent=true is required to store preferences"}
    if not isinstance(prefs, dict):
        return {"ok": False, "error": "prefs must be an object"}

    cleaned: Dict[str, str] = {}
    for key, allowed in _ALLOWED.items():
        if key not in prefs:
            continue
        val = str(prefs.get(key) or "").strip()[:80]
        if not val:
            continue
        if allowed is not None and val.lower() not in allowed:
            return {
                "ok": False,
                "error": f"invalid {key}; allowed: {sorted(allowed)}",
            }
        cleaned[key] = val.lower() if allowed is not None else val

    with _lock:
        data = _load()
        existing = data.get(owner_id) or {}
        merged = dict(existing.get("prefs") or {})
        merged.update(cleaned)
        data[owner_id] = {
            "prefs": merged,
            "consent": True,
            "updated_at": _now(),
        }
        _save(data)
    return {"ok": True, "prefs": merged, "updated_at": data[owner_id]["updated_at"]}


def clear_preferences(owner_id: str) -> Dict[str, Any]:
    owner_id = (owner_id or "guest").strip()[:120]
    with _lock:
        data = _load()
        if owner_id in data:
            del data[owner_id]
            _save(data)
            return {"ok": True, "cleared": True}
    return {"ok": True, "cleared": False}


def format_preference_context(owner_id: str) -> Optional[str]:
    """Return a short system-context string for injection, or None."""
    info = get_preferences(owner_id)
    prefs = info.get("prefs") or {}
    if not prefs or not info.get("consent"):
        return None
    bits = []
    if prefs.get("tone"):
        bits.append(f"preferred tone: {prefs['tone']}")
    if prefs.get("verbosity"):
        bits.append(f"verbosity: {prefs['verbosity']}")
    if prefs.get("language") or prefs.get("reply_language"):
        bits.append(
            f"reply language preference: {prefs.get('reply_language') or prefs.get('language')}"
        )
    if prefs.get("code_style"):
        bits.append(f"code style: {prefs['code_style']}")
    if not bits:
        return None
    return (
        "User style preferences (consented; follow when it improves the answer, "
        "never override safety or truth): " + "; ".join(bits)
    )
