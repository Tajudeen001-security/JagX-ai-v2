"""User-scoped Supabase tools for JagX AI.

Every request must carry a Supabase access token. The token is validated by
Supabase Auth, and all data requests use that same user token so database RLS
remains authoritative. No service-role key or arbitrary SQL is used.
"""
from __future__ import annotations

import os
from typing import Any

import requests
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/supabase", tags=["supabase-user-tools"])


def _config() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    public_key = (os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_PUBLISHABLE_KEY") or "").strip()
    if not url or not public_key:
        raise HTTPException(status_code=503, detail="Supabase integration is not configured")
    return url, public_key


def _user_context(authorization: str | None) -> tuple[str, str, str, str]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="A signed-in Supabase session is required")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="A signed-in Supabase session is required")
    url, public_key = _config()
    try:
        response = requests.get(
            f"{url}/auth/v1/user",
            headers={"apikey": public_key, "Authorization": f"Bearer {token}"},
            timeout=8,
        )
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Could not validate the Supabase session")
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Supabase session is invalid or expired")
    user = response.json()
    user_id = user.get("id") if isinstance(user, dict) else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Supabase did not return a valid user")
    return url, public_key, token, str(user_id)


class MessageSave(BaseModel):
    chat_id: str = Field(min_length=1, max_length=160)
    payload: dict[str, Any]
    updated_at: str | None = None


@router.get("/me")
def supabase_me(authorization: str | None = Header(default=None)):
    url, public_key, token, _user_id = _user_context(authorization)
    try:
        response = requests.get(
            f"{url}/auth/v1/user",
            headers={"apikey": public_key, "Authorization": f"Bearer {token}"},
            timeout=8,
        )
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Could not retrieve the Supabase user")
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Supabase session is invalid or expired")
    user = response.json()
    return {"ok": True, "user": {"id": user.get("id"), "email": user.get("email")}}


@router.get("/messages")
def list_user_messages(authorization: str | None = Header(default=None), limit: int = 50):
    url, public_key, token, _user_id = _user_context(authorization)
    limit = max(1, min(limit, 100))
    try:
        response = requests.get(
            f"{url}/rest/v1/messages",
            params={"select": "chat_id,payload,updated_at", "order": "updated_at.desc", "limit": str(limit)},
            headers={"apikey": public_key, "Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Supabase messages are temporarily unavailable")
    if response.status_code >= 400:
        # Keep database/schema details out of the public API response.
        raise HTTPException(status_code=403 if response.status_code in (401, 403) else 502,
                            detail="Messages could not be read; check your account permissions")
    return {"ok": True, "messages": response.json()}


@router.put("/messages")
def save_user_message(body: MessageSave, authorization: str | None = Header(default=None)):
    url, public_key, token, user_id = _user_context(authorization)
    row = {"user_id": user_id, "chat_id": body.chat_id, "payload": body.payload}
    if body.updated_at:
        row["updated_at"] = body.updated_at
    try:
        response = requests.post(
            f"{url}/rest/v1/messages",
            params={"on_conflict": "user_id,chat_id"},
            json=row,
            headers={
                "apikey": public_key,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            timeout=10,
        )
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Supabase messages are temporarily unavailable")
    if response.status_code >= 400:
        raise HTTPException(status_code=403 if response.status_code in (401, 403) else 502,
                            detail="Message could not be saved; check your account permissions")
    return {"ok": True, "saved": True}
