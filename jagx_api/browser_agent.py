"""Lightweight research browser agent (no headless Chrome).

Capabilities:
- fetch_url: single page text extract
- browse_research: search + follow top result pages + combined extract

Not a full interactive browser (no click/type/JS). Safe for production.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

from jagx_api.providers import HTTP
from jagx_api.tools import free_web_search

logger = logging.getLogger("jagx-ai")

_MAX_CHARS = 10000
_ALLOWED = {"http", "https"}
_UA = "JagXAI-Browser/9.9 (+https://jagxai.name.ng; research agent)"


def _ok_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in _ALLOWED and bool(p.netloc)
    except Exception:
        return False


def _strip_html(body: str) -> str:
    body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", body)
    body = re.sub(r"(?is)<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body


def fetch_url(url: str, max_chars: int = _MAX_CHARS) -> str:
    url = (url or "").strip()
    if not url:
        return "empty url"
    if not _ok_url(url):
        return "only public http/https URLs are allowed"
    try:
        r = HTTP.get(
            url,
            headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml,text/plain,application/json"},
            timeout=15,
            follow_redirects=True,
        )
        if r.status_code >= 400:
            return f"fetch failed ({r.status_code})"
        ctype = (r.headers.get("content-type") or "").lower()
        body = r.text or ""
        if "application/json" in ctype:
            return body[:max_chars]
        text = _strip_html(body)
        title_m = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
        title = _strip_html(title_m.group(1)) if title_m else ""
        if title:
            text = f"Title: {title}\n\n{text}"
        if not text.strip():
            return "no readable text extracted"
        return text[:max_chars]
    except Exception as exc:
        logger.warning("fetch_url failed: %s", type(exc).__name__)
        return f"fetch error: {type(exc).__name__}"


def _extract_links(html: str, base: str, limit: int = 8) -> list[str]:
    links = []
    for href in re.findall(r'(?is)<a[^>]+href=["\']([^"\']+)["\']', html or ""):
        href = href.strip()
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        full = urljoin(base, href)
        if _ok_url(full) and full not in links:
            links.append(full)
        if len(links) >= limit:
            break
    return links


def _urls_from_search_blob(blob: str, limit: int = 4) -> list[str]:
    urls = []
    for m in re.findall(r"https?://[^\s\)\]\>\"']+", blob or ""):
        u = m.rstrip(".,;:")
        if _ok_url(u) and "duckduckgo.com" not in u and u not in urls:
            urls.append(u)
        if len(urls) >= limit:
            break
    return urls


def browse_research(query: str, max_pages: int = 3) -> str:
    """Search the web, open top result pages, return combined extracts (untrusted data)."""
    q = (query or "").strip()[:220]
    if not q:
        return "empty query"
    max_pages = max(1, min(int(max_pages or 3), 4))

    search_blob = free_web_search(q, max_results=6)
    urls = _urls_from_search_blob(search_blob, limit=max_pages)

    parts = [f"## Search summary for: {q}\n{search_blob[:3500]}"]

    for i, url in enumerate(urls, 1):
        try:
            r = HTTP.get(
                url,
                headers={"User-Agent": _UA, "Accept": "text/html,text/plain"},
                timeout=12,
                follow_redirects=True,
            )
            if r.status_code >= 400:
                parts.append(f"\n## Page {i}: {url}\nfetch failed ({r.status_code})")
                continue
            text = _strip_html(r.text or "")[:4500]
            title_m = re.search(r"(?is)<title[^>]*>(.*?)</title>", r.text or "")
            title = _strip_html(title_m.group(1)) if title_m else url
            parts.append(f"\n## Page {i}: {title}\nURL: {url}\n{text}")
        except Exception as exc:
            parts.append(f"\n## Page {i}: {url}\nerror: {type(exc).__name__}")

    if not urls:
        parts.append("\n(No follow-up pages opened — search snippets only.)")

    out = "\n".join(parts)
    return out[:18000]


def run_browser_tool(name: str, inp: dict) -> tuple[str, Optional[Any]]:
    """Dispatch browser-related tools. Returns (text, attachment)."""
    name = (name or "").strip()
    inp = inp if isinstance(inp, dict) else {}
    if name == "fetch_url":
        return fetch_url(str(inp.get("url") or "")), None
    if name in ("browse_research", "browser_research", "research_browse"):
        return browse_research(str(inp.get("query") or ""), int(inp.get("max_pages") or 3)), None
    return f"unknown browser tool: {name}", None
