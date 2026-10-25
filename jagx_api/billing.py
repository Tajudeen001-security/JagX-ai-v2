"""Manual payment + admin-confirmed tier upgrades.

Flow:
  1. User pays to OPay account (see SETTLEMENT_ACCOUNT)
  2. User submits upgrade request with payment reference
  3. Admin confirms with admin_secret → API key tier is upgraded.

Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from jagx_api.config import settings

logger = logging.getLogger("jagx-ai")
_lock = threading.RLock()

ADMIN_EMAILS = {
    "gbadamositajudeenwan@gmail.com",
    "gbadamositajudeeneh@gmail.com",
}

SETTLEMENT_ACCOUNT = {
    "bank": "OPay",
    "account_number": "9160654415",
    "account_name": "Gbadamosi Tajudeen Olajide",
    "currency": "NGN",
}

TIER_CATALOG = {
    "free": {
        "label": "Free",
        "price_ngn": 0,
        "hourly_limit": 100,
        "priority": 10,
        "features": ["chat", "tools", "public rate limits"],
    },
    "premium": {
        "label": "Premium",
        "price_ngn": 15000,
        "hourly_limit": 400,
        "priority": 50,
        "features": ["priority jobs", "higher limits", "multi-agent"],
    },
    "premium_plus": {
        "label": "Premium Plus",
        "price_ngn": 35000,
        "hourly_limit": 1000,
        "priority": 80,
        "features": ["top priority", "highest limits", "reason mode jobs"],
    },
}

VALID_PAID_TIERS = {"premium", "premium_plus"}


def _path() -> Path:
    return Path(getattr(settings, "billing_file", None) or "jagx_billing.json")


def _load() -> dict:
    p = _path()
    if not p.exists():
        return {"requests": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"requests": {}}
        data.setdefault("requests", {})
        return data
    except Exception:
        return {"requests": {}}


def _save(data: dict) -> None:
    _path().write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def catalog() -> dict:
    return {
        "currency": "NGN",
        "settlement": "manual",
        "settlement_account": SETTLEMENT_ACCOUNT,
        "admin_emails": sorted(ADMIN_EMAILS),
        "tiers": TIER_CATALOG,
        "instructions": (
            f"1) Transfer the tier amount to OPay {SETTLEMENT_ACCOUNT['account_number']} "
            f"({SETTLEMENT_ACCOUNT['account_name']}). "
            "2) Use GET /billing/payment-note?tier=premium for a ready transfer note. "
            "3) Submit POST /billing/upgrade-request with payment_ref. "
            "4) Admin confirms at /billing/admin/panel."
        ),
        "admin_panel": "/billing/admin/panel",
    }


def payment_note_template(tier: str = "premium") -> dict:
    """Copy-paste transfer note for bank/OPay narration."""
    tier = (tier or "premium").lower().strip()
    if tier not in VALID_PAID_TIERS:
        return {"ok": False, "error": "tier must be premium or premium_plus"}
    info = TIER_CATALOG[tier]
    amount = info["price_ngn"]
    acct = SETTLEMENT_ACCOUNT
    narration = f"JagX {info['label']} upgrade"
    text = (
        f"Pay ₦{amount:,} to:\n"
        f"Bank: {acct['bank']}\n"
        f"Account: {acct['account_number']}\n"
        f"Name: {acct['account_name']}\n"
        f"Narration: {narration}\n\n"
        f"After payment, submit /billing/upgrade-request with your transfer reference."
    )
    return {
        "ok": True,
        "tier": tier,
        "amount_ngn": amount,
        "settlement_account": acct,
        "narration": narration,
        "text": text,
    }


def is_admin_email(email: str) -> bool:
    return (email or "").strip().lower() in ADMIN_EMAILS


def create_upgrade_request(
    *,
    api_key: str,
    owner: str,
    target_tier: str,
    payment_ref: str,
    amount_ngn: Optional[float] = None,
    note: str = "",
    payer_email: str = "",
    payer_name: str = "",
) -> dict:
    target_tier = (target_tier or "").lower().strip()
    if target_tier not in VALID_PAID_TIERS:
        return {"ok": False, "error": "target_tier must be premium or premium_plus"}
    payment_ref = (payment_ref or "").strip()[:200]
    if len(payment_ref) < 3:
        return {"ok": False, "error": "payment_ref is required (bank ref / transfer note)"}

    expected = TIER_CATALOG[target_tier]["price_ngn"]
    rid = str(uuid.uuid4())
    now = time.time()
    row = {
        "id": rid,
        "api_key_hint": (api_key[:12] + "…") if api_key else "",
        "api_key": api_key,
        "owner": (owner or "")[:120],
        "target_tier": target_tier,
        "payment_ref": payment_ref,
        "amount_ngn": float(amount_ngn) if amount_ngn is not None else expected,
        "expected_ngn": expected,
        "note": (note or "")[:500],
        "payer_email": (payer_email or "")[:120],
        "payer_name": (payer_name or "")[:120],
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "confirmed_by": None,
        "confirmed_at": None,
    }
    with _lock:
        data = _load()
        data["requests"][rid] = row
        _save(data)
    return {"ok": True, "request": _public_view(row)}


def _public_view(row: dict) -> dict:
    return {
        "id": row["id"],
        "owner": row.get("owner"),
        "target_tier": row.get("target_tier"),
        "payment_ref": row.get("payment_ref"),
        "amount_ngn": row.get("amount_ngn"),
        "expected_ngn": row.get("expected_ngn"),
        "status": row.get("status"),
        "note": row.get("note"),
        "payer_email": row.get("payer_email"),
        "payer_name": row.get("payer_name"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "confirmed_by": row.get("confirmed_by"),
        "confirmed_at": row.get("confirmed_at"),
        "api_key_hint": row.get("api_key_hint"),
    }


def list_requests(*, status: Optional[str] = None, limit: int = 50) -> list:
    with _lock:
        data = _load()
        rows = list(data.get("requests", {}).values())
    if status:
        rows = [r for r in rows if r.get("status") == status]
    rows.sort(key=lambda r: float(r.get("created_at") or 0), reverse=True)
    return [_public_view(r) for r in rows[: max(1, min(limit, 200))]]


def get_request(request_id: str) -> Optional[dict]:
    with _lock:
        row = _load().get("requests", {}).get(request_id)
    return _public_view(row) if row else None


def confirm_request(
    request_id: str,
    *,
    admin_secret: str,
    admin_email: str,
    action: str = "confirm",
) -> dict:
    if not settings.admin_secret or admin_secret != settings.admin_secret:
        return {"ok": False, "error": "invalid admin_secret"}
    email = (admin_email or "").strip().lower()
    if not is_admin_email(email):
        return {
            "ok": False,
            "error": "admin_email not authorized",
            "allowed": sorted(ADMIN_EMAILS),
        }
    action = (action or "confirm").lower()
    if action not in ("confirm", "reject"):
        return {"ok": False, "error": "action must be confirm or reject"}

    with _lock:
        data = _load()
        row = data.get("requests", {}).get(request_id)
        if not row:
            return {"ok": False, "error": "request not found"}
        if row.get("status") != "pending":
            return {"ok": False, "error": f"request already {row.get('status')}"}

        now = time.time()
        if action == "reject":
            row["status"] = "rejected"
            row["updated_at"] = now
            row["confirmed_by"] = email
            row["confirmed_at"] = now
            data["requests"][request_id] = row
            _save(data)
            return {"ok": True, "request": _public_view(row)}

        api_key = row.get("api_key") or ""
        target = row.get("target_tier")
        from jagx_api import auth

        upgraded = auth.set_tier(api_key, target)
        if not upgraded.get("ok"):
            return {"ok": False, "error": upgraded.get("error") or "tier update failed"}

        row["status"] = "confirmed"
        row["updated_at"] = now
        row["confirmed_by"] = email
        row["confirmed_at"] = now
        data["requests"][request_id] = row
        _save(data)

    return {
        "ok": True,
        "request": _public_view(row),
        "tier": target,
        "message": f"API key upgraded to {target}",
    }
