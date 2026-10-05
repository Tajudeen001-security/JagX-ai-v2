"""API key management and rate limiting."""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from jagx_api.config import settings

_lock = threading.Lock()
_rate: dict = defaultdict(list)
_ip_rate: dict = defaultdict(list)


def _load_keys() -> dict:
    path = Path(settings.keys_file)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_keys(keys: dict) -> None:
    Path(settings.keys_file).write_text(
        json.dumps(keys, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def create_key(owner_label: str, tier: str = "free", admin_secret: str = "") -> dict:
    if admin_secret != settings.admin_secret:
        return {"ok": False, "error": "invalid admin_secret"}
    tier = (tier or "free").lower()
    if tier not in settings.tier_hourly:
        tier = "free"
    key = "jagx-" + secrets.token_hex(16)
    with _lock:
        keys = _load_keys()
        keys[key] = {
            "owner": (owner_label or "anonymous")[:120],
            "tier": tier,
            "created_at": time.time(),
            "never_expires": True,
            "requests": 0,
        }
        _save_keys(keys)
    return {
        "ok": True,
        "api_key": key,
        "owner": owner_label,
        "tier": tier,
        "hourly_limit": settings.tier_hourly.get(tier),
        "never_expires": True,
    }


def is_valid_key(key: str) -> bool:
    if not key:
        return False
    if key in settings.permanent_keys:
        return True
    with _lock:
        return key in _load_keys()


def key_info(key: str) -> Optional[dict]:
    if key in settings.permanent_keys:
        return {
            "owner": "permanent",
            "tier": "admin",
            "never_expires": True,
            "hourly_limit": None,
        }
    with _lock:
        meta = _load_keys().get(key)
    if not meta:
        return None
    return {
        "owner": meta.get("owner"),
        "tier": meta.get("tier", "free"),
        "never_expires": True,
        "hourly_limit": settings.tier_hourly.get(meta.get("tier", "free")),
        "requests": meta.get("requests", 0),
    }


def check_rate_limit(key: str) -> tuple:
    info = key_info(key)
    if not info:
        return False, "invalid key"
    limit = info.get("hourly_limit")
    if limit is None:
        return True, "unlimited"
    now = time.time()
    with _lock:
        stamps = [t for t in _rate[key] if now - t < 3600]
        if len(stamps) >= int(limit):
            return False, f"hourly limit reached ({limit}/hour)"
        stamps.append(now)
        _rate[key] = stamps
        keys = _load_keys()
        if key in keys:
            keys[key]["requests"] = int(keys[key].get("requests") or 0) + 1
            _save_keys(keys)
    return True, f"{len(stamps)}/{limit} this hour"


def check_ip_rate(ip: str) -> bool:
    now = time.time()
    with _lock:
        stamps = [t for t in _ip_rate[ip] if now - t < 60]
        if len(stamps) >= settings.global_ip_rpm:
            return False
        stamps.append(now)
        _ip_rate[ip] = stamps
    return True
