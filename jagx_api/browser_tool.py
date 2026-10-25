"""Lightweight URL fetch for Researcher (not a full browser agent)."""
from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from jagx_api.providers import HTTP

logger = logging.getLogger("jagx-ai")

_MAX_CHARS = 12000
_ALLOWED_SCHEMES = {"http", "https"}


def fetch_url(url: str, max_chars: int = _MAX_CHARS) -> str:
    """Fetch a public URL and return plain-text extract. Untrusted data only."""
    url = (url or "").strip()
    if not url:
        return "empty url"
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return "only http/https URLs are allowed"
    if not parsed.netloc:
        return "invalid url"
    try:
        headers = {
            "User-Agent": "JagXAI-Fetch/9.7 (+https://jagxai.name.ng; research bot)",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json",
        }
        r = HTTP.get(url, headers=headers, timeout=15, follow_redirects=True)
        if r.status_code >= 400:
            return f"fetch failed ({r.status_code})"
        ctype = (r.headers.get("content-type") or "").lower()
        body = r.text or ""
        if "application/json" in ctype:
            return body[:max_chars]
        body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
        body = re.sub(r"(?is)<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            return "no readable text extracted"
        return body[:max_chars]
    except Exception as exc:
        logger.warning("fetch_url failed: %s", type(exc).__name__)
        return f"fetch error: {type(exc).__name__}"
