"""JagX Mail delivery API — calls the real JagX Mail server, never Resend directly."""
from __future__ import annotations

import logging
import os
import re
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
    # Authenticate the JagX AI caller using its own API key and rate limits.
    key = (x_api_key or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok, info = auth.check_rate_limit(key)
    if not ok:
        raise HTTPException(status_code=429, detail=info)

    # The JagX Mail service credential is server-only and is never returned to callers.
    mail_url = os.environ.get(
        "JAGX_MAIL_API_URL",
        "https://jagx-mail-jagxs-projects.vercel.app/api/v1/agent/send",
    ).strip()
    mail_key = os.environ.get("JAGX_MAIL_API_KEY", "").strip()
    if not mail_key:
        raise HTTPException(status_code=503, detail="JagX Mail integration is not configured")
    if not mail_url.startswith("https://"):
        raise HTTPException(status_code=503, detail="JagX Mail API URL must use HTTPS")

    recipients = [addr.strip() for addr in req.to]
    if any(not EMAIL_RE.fullmatch(addr) for addr in recipients):
        raise HTTPException(status_code=400, detail="One or more recipient addresses are invalid")

    sent = []
    failed = []
    for recipient in recipients:
        payload = {"to": recipient, "subject": req.subject, "text": req.text}
        if req.html:
            payload["html"] = req.html
        try:
            response = requests.post(
                mail_url,
                headers={"Authorization": f"Bearer {mail_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=20,
            )
        except requests.RequestException:
            logger.warning("JagX Mail request failed")
            failed.append({"to": recipient, "reason": "JagX Mail could not be reached"})
            continue

        if response.status_code not in (200, 202):
            # Do not forward provider details or secrets from the downstream service.
            failed.append({"to": recipient, "reason": "JagX Mail rejected the message"})
            continue
        try:
            result = response.json()
        except ValueError:
            result = {}
        sent.append({"to": recipient, "messageId": result.get("messageId"), "status": result.get("status", "accepted")})

    if failed and not sent:
        raise HTTPException(status_code=502, detail={"message": "JagX Mail rejected all messages", "failed": failed})
    return {
        "ok": not failed,
        "provider": "JagX Mail",
        "accepted": len(sent),
        "results": sent,
        "failed": failed,
        "status": "accepted_by_jagx_mail" if not failed else "partially_accepted",
    }
