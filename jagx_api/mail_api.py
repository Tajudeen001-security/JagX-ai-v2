"""JagX Mail delivery API — credentials stay server-side."""
from __future__ import annotations

import os
import re
import logging
from typing import Optional

import requests
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from jagx_api import auth

logger = logging.getLogger("jagx-mail")
router = APIRouter(prefix="/mail", tags=["mail"])
EMAIL_RE = re.compile(r"^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$")


class SendMailRequest(BaseModel):
    to: list[str] = Field(..., min_length=1, max_length=10)
    subject: str = Field(..., min_length=1, max_length=180)
    text: str = Field(..., min_length=1, max_length=20000)
    html: Optional[str] = Field(default=None, max_length=40000)


@router.post("/send")
def send_mail(req: SendMailRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    key = (x_api_key or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok, info = auth.check_rate_limit(key)
    if not ok:
        raise HTTPException(status_code=429, detail=info)

    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Mail delivery is not configured")
    sender = os.environ.get("JAGX_MAIL_FROM", "JagX AI <no-reply@jagxai.name.ng>").strip()
    if not re.search(r"@(?:[A-Za-z0-9-]+\.)*jagxai\.name\.ng\s*>?$", sender, re.I):
        raise HTTPException(status_code=503, detail="Configured sender must use the verified JagX domain")
    recipients = [addr.strip() for addr in req.to]
    if any(not EMAIL_RE.fullmatch(addr) for addr in recipients):
        raise HTTPException(status_code=400, detail="One or more recipient addresses are invalid")

    payload = {"from": sender, "to": recipients, "subject": req.subject, "text": req.text}
    if req.html:
        payload["html"] = req.html
    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
    except requests.RequestException:
        logger.warning("Resend request failed")
        raise HTTPException(status_code=502, detail="Mail provider could not be reached")
    if response.status_code >= 400:
        logger.warning("Resend rejected mail request with status %s", response.status_code)
        raise HTTPException(status_code=502, detail="Mail provider rejected the message")
    try:
        result = response.json()
    except ValueError:
        result = {}
    return {"ok": True, "id": result.get("id"), "provider": "Resend"}
