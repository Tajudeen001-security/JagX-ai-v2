"""
JagX AI V9.6 — Preferences API (consented style profile).
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from jagx_api import auth
from jagx_api import preferences as prefs

logger = logging.getLogger("jagx-ai")
router = APIRouter(prefix="/preferences", tags=["preferences"])


def _owner(x_api_key: Optional[str]) -> str:
    key = (x_api_key or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok, info = auth.check_rate_limit(key)
    if not ok:
        raise HTTPException(status_code=429, detail=info)
    return hashlib.sha256(("jagx-prefs-v1:" + key).encode()).hexdigest()


class PrefsBody(BaseModel):
    tone: Optional[str] = Field(default=None, max_length=40)
    verbosity: Optional[str] = Field(default=None, max_length=40)
    language: Optional[str] = Field(default=None, max_length=40)
    reply_language: Optional[str] = Field(default=None, max_length=40)
    code_style: Optional[str] = Field(default=None, max_length=40)
    consent: bool = False


@router.get("")
def get_prefs(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    owner = _owner(x_api_key)
    return prefs.get_preferences(owner)


@router.post("")
def set_prefs(body: PrefsBody, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    owner = _owner(x_api_key)
    payload: Dict[str, str] = {}
    for key in ("tone", "verbosity", "language", "reply_language", "code_style"):
        val = getattr(body, key, None)
        if val:
            payload[key] = val
    result = prefs.set_preferences(owner, payload, consent=body.consent)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "invalid preferences"))
    return result


@router.delete("")
def clear_prefs(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    owner = _owner(x_api_key)
    return prefs.clear_preferences(owner)
