"""
JagX secret vault — per-user tokens (GitHub, etc.).
Stored in a separate file, never included in job logs or API responses.
Used only when a job needs to write code or call a connector.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("jagx-ai")
VAULT_FILE = os.environ.get("JAGX_VAULT_FILE", "jagx_vault.json")
_lock = threading.Lock()


def _load() -> dict:
    if not os.path.exists(VAULT_FILE):
        return {}
    try:
        with open(VAULT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    with open(VAULT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    try:
        os.chmod(VAULT_FILE, 0o600)
    except Exception:
        pass


def put_secret(user_id: str, key: str, value: str) -> dict:
    user_id = (user_id or "").strip()[:120]
    key = (key or "").strip()[:40]
    value = (value or "").strip()
    if not user_id or not key or not value:
        return {"ok": False, "error": "user_id, key, value required"}
    if key not in ("github", "openrouter", "supabase"):
        return {"ok": False, "error": "unsupported key"}
    with _lock:
        data = _load()
        row = data.get(user_id) or {}
        row[key] = value
        data[user_id] = row
        _save(data)
    return {"ok": True, "stored": key, "user_id": user_id}


def get_secret(user_id: str, key: str) -> Optional[str]:
    """Internal only. Never return this from an HTTP response."""
    with _lock:
        data = _load()
    row = data.get((user_id or "").strip()) or {}
    val = row.get(key)
    return val if isinstance(val, str) and val else None


def has_secret(user_id: str, key: str) -> bool:
    return bool(get_secret(user_id, key))


def delete_secret(user_id: str, key: str) -> dict:
    with _lock:
        data = _load()
        row = data.get(user_id) or {}
        row.pop(key, None)
        data[user_id] = row
        _save(data)
    return {"ok": True}


def status(user_id: str) -> dict:
    """Safe status — names only, no values."""
    with _lock:
        row = _load().get((user_id or "").strip()) or {}
    return {
        "user_id": user_id,
        "github": bool(row.get("github")),
        "openrouter": bool(row.get("openrouter")),
        "supabase": bool(row.get("supabase")),
    }


def register_vault_routes(app=None) -> None:
    app = app or globals().get("app")
    if app is None:
        return
    from fastapi import Body

    @app.post("/vault")
    def vault_put(body: dict = Body(...)):
        return put_secret(
            str(body.get("user_id") or ""),
            str(body.get("key") or ""),
            str(body.get("value") or ""),
        )

    @app.get("/vault/status")
    def vault_status(user_id: str = ""):
        return status(user_id)

    @app.post("/vault/delete")
    def vault_del(body: dict = Body(...)):
        return delete_secret(str(body.get("user_id") or ""), str(body.get("key") or ""))

    logger.info("JagX vault routes registered (values never returned)")
