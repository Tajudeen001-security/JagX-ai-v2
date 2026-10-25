"""Billing / manual tier upgrade API."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from jagx_api import auth, billing
from jagx_api.config import settings

router = APIRouter(prefix="/billing", tags=["billing"])


class UpgradeRequest(BaseModel):
    target_tier: str = Field(..., description="premium | premium_plus")
    payment_ref: str = Field(..., min_length=3, max_length=200)
    amount_ngn: Optional[float] = None
    note: str = ""
    payer_email: str = ""
    payer_name: str = ""


class AdminConfirmRequest(BaseModel):
    request_id: str
    admin_secret: str
    admin_email: str
    action: str = Field(default="confirm", pattern="^(confirm|reject)$")


def _require_key(x_api_key: Optional[str]) -> str:
    key = (x_api_key or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok, info = auth.check_rate_limit(key)
    if not ok:
        raise HTTPException(status_code=429, detail=info)
    return key


@router.get("/catalog")
def billing_catalog():
    return {"ok": True, **billing.catalog()}


@router.post("/upgrade-request")
def upgrade_request(
    req: UpgradeRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    key = _require_key(x_api_key)
    info = auth.key_info(key) or {}
    result = billing.create_upgrade_request(
        api_key=key,
        owner=str(info.get("owner") or ""),
        target_tier=req.target_tier,
        payment_ref=req.payment_ref,
        amount_ngn=req.amount_ngn,
        note=req.note,
        payer_email=req.payer_email,
        payer_name=req.payer_name,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "bad request"))
    return result


@router.get("/my-requests")
def my_requests(
    limit: int = 20,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    key = _require_key(x_api_key)
    info = auth.key_info(key) or {}
    owner = str(info.get("owner") or "")
    all_rows = billing.list_requests(limit=200)
    mine = [r for r in all_rows if r.get("owner") == owner][: max(1, min(limit, 50))]
    return {"ok": True, "requests": mine, "current_tier": info.get("tier")}


@router.get("/admin/pending")
def admin_pending(
    admin_secret: str,
    admin_email: str,
    limit: int = 50,
):
    if not settings.admin_secret or admin_secret != settings.admin_secret:
        raise HTTPException(status_code=403, detail="invalid admin_secret")
    if not billing.is_admin_email(admin_email):
        raise HTTPException(status_code=403, detail="admin_email not authorized")
    return {"ok": True, "requests": billing.list_requests(status="pending", limit=limit)}


@router.post("/admin/confirm")
def admin_confirm(req: AdminConfirmRequest):
    result = billing.confirm_request(
        req.request_id,
        admin_secret=req.admin_secret,
        admin_email=req.admin_email,
        action=req.action,
    )
    if not result.get("ok"):
        err = str(result.get("error", "failed")).lower()
        code = 403 if "admin" in err or "secret" in err else 400
        raise HTTPException(status_code=code, detail=result.get("error", "failed"))
    return result
