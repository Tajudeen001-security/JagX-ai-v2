"""
JagX MCP proxy — remote MCP + built-in free tools.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger("jagx-ai")

try:
    HTTP  # type: ignore  # noqa: F821
except NameError:
    HTTP = requests.Session()

UA = os.environ.get("JAGX_OSM_USER_AGENT", "JagXAI/7.1 (mcp-proxy)")


def _builtin_news(topic: str = "") -> str:
    try:
        import jagx_extensions as jx
        return jx.fetch_news(topic or "")
    except Exception:
        return _ddg(f"{topic} news".strip() or "world news")


def _builtin_geo(q: str) -> str:
    try:
        import jagx_extensions as jx
        return jx.geocode_place(q or "")
    except Exception as e:
        return f"Maps unavailable: {e}"


def _builtin_weather(place: str) -> str:
    try:
        import jagx_extensions as jx
        return jx.weather_for_place(place or "")
    except Exception as e:
        return f"Weather unavailable: {e}"


def _wiki(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Give a Wikipedia topic."
    try:
        r = HTTP.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(q.replace(" ", "_")),
            timeout=12,
            headers={"User-Agent": UA},
        )
        if r.status_code != 200:
            return f"No Wikipedia page for “{q}”."
        data = r.json()
        title = data.get("title") or q
        extract = (data.get("extract") or "")[:1200]
        return f"{title} (Wikipedia)\n{extract}"
    except Exception as e:
        return f"Wikipedia error: {e}"


def _ddg(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Give a search query."
    try:
        r = HTTP.get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=12,
            headers={"User-Agent": UA},
        )
        data = r.json() if r.status_code == 200 else {}
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:800])
        for item in (data.get("RelatedTopics") or [])[:5]:
            if isinstance(item, dict) and item.get("Text"):
                parts.append("- " + item["Text"][:200])
        if parts:
            return "Web search:\n" + "\n".join(parts)
        return f"No instant answer for “{q}”. Try https://duckduckgo.com/?q={quote(q)}"
    except Exception as e:
        return f"Search error: {e}"


BUILTIN = {
    "jagx_news": lambda a: _builtin_news(str(a.get("topic") or a.get("query") or "")),
    "jagx_maps": lambda a: _builtin_geo(str(a.get("query") or a.get("place") or "")),
    "jagx_weather": lambda a: _builtin_weather(str(a.get("place") or a.get("query") or "")),
    "wikipedia": lambda a: _wiki(str(a.get("query") or a.get("topic") or "")),
    "jagx_web": lambda a: _ddg(str(a.get("query") or "")),
    "web": lambda a: _ddg(str(a.get("query") or "")),
}


def _mcp_headers(token: Optional[str] = None) -> Dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": UA,
    }
    if token:
        h["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    return h


def mcp_rpc(url: str, method: str, params: Optional[dict] = None, token: Optional[str] = None, timeout: int = 25) -> dict:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    r = HTTP.post(url, json=payload, headers=_mcp_headers(token), timeout=timeout)
    text = r.text or ""
    if "event-stream" in r.headers.get("content-type", ""):
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except Exception:
                    continue
        return {"error": {"message": f"SSE parse failed ({r.status_code})"}}
    try:
        return r.json()
    except Exception:
        return {"error": {"message": f"HTTP {r.status_code}: {text[:300]}"}}


def mcp_list_tools(url: str, token: Optional[str] = None) -> dict:
    if not url:
        return {"ok": False, "error": "Missing MCP URL"}
    try:
        mcp_rpc(
            url,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "JagX-AI", "version": "7.1"},
            },
            token=token,
            timeout=12,
        )
    except Exception:
        pass
    try:
        res = mcp_rpc(url, "tools/list", {}, token=token)
        if "error" in res:
            return {"ok": False, "error": res["error"]}
        tools = (res.get("result") or {}).get("tools") or []
        return {"ok": True, "tools": tools, "count": len(tools)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def mcp_call_tool(url: str, name: str, arguments: Optional[dict] = None, token: Optional[str] = None) -> dict:
    if not url or not name:
        return {"ok": False, "error": "url and name required"}
    try:
        res = mcp_rpc(
            url,
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            token=token,
            timeout=45,
        )
        if "error" in res:
            return {"ok": False, "error": res["error"]}
        result = res.get("result") or res
        content = result.get("content") if isinstance(result, dict) else None
        if isinstance(content, list):
            texts = [str(c.get("text") if isinstance(c, dict) else c) for c in content]
            return {"ok": True, "text": "\n".join(texts)}
        return {"ok": True, "text": json.dumps(result, ensure_ascii=False)[:8000]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _pick_builtin(message: str, enabled_ids: List[str]):
    m = (message or "").lower()
    ids = set(enabled_ids or [])

    def has(*names):
        return (not ids) or any(n in ids for n in names)

    if has("jagx_news") and re.search(r"\b(news|headline|breaking)\b", m):
        topic = next((w for w in ("nigeria", "africa", "world", "tech") if w in m), "")
        return ("jagx_news", {"topic": topic})
    if has("jagx_weather", "jagx_maps") and re.search(r"\b(weather|temperature|forecast)\b", m):
        place = re.sub(r".*\b(?:in|for|at)\s+", "", message, flags=re.I).strip()[:80] or "Lagos"
        return ("jagx_weather", {"place": place})
    if has("jagx_maps") and re.search(r"\b(where is|map of|locate|geocode)\b", m):
        q = re.sub(r".*\b(?:where is|map of|locate|geocode)\s+", "", message, flags=re.I).strip()[:100]
        return ("jagx_maps", {"query": q or message})
    if has("wikipedia") and re.search(r"\b(wikipedia|who is|what is)\b", m):
        q = re.sub(r".*\b(?:wikipedia|who is|what is)\s+", "", message, flags=re.I).strip()[:100]
        return ("wikipedia", {"query": q or message})
    if has("jagx_web", "web") and re.search(r"\b(search|google|look up)\b", m):
        q = re.sub(r".*\b(?:search|google|look up)\s+(?:for\s+)?", "", message, flags=re.I).strip()[:120]
        return ("jagx_web", {"query": q or message})
    return None


def run_connectors_for_message(message: str, connectors: List[dict]) -> dict:
    enabled = [c for c in (connectors or []) if c.get("enabled", True)]
    ids = [str(c.get("id") or "") for c in enabled]
    pick = _pick_builtin(message, ids)
    if pick:
        tool_id, args = pick
        fn = BUILTIN.get(tool_id)
        if fn:
            try:
                return {"ok": True, "source": "builtin", "tool": tool_id, "text": fn(args)}
            except Exception as e:
                return {"ok": False, "tool": tool_id, "error": str(e)}
    for c in enabled:
        url = (c.get("url") or "").strip()
        if not url:
            continue
        name = (c.get("name") or c.get("id") or "").lower()
        if name and name.split()[0] not in message.lower() and str(c.get("id")) not in message.lower():
            continue
        token = c.get("token") or c.get("auth_token")
        listed = mcp_list_tools(url, token=token)
        if not listed.get("ok"):
            continue
        tools = listed.get("tools") or []
        if not tools:
            continue
        t0 = tools[0]
        tname = t0.get("name") if isinstance(t0, dict) else str(t0)
        called = mcp_call_tool(url, tname, {"query": message}, token=token)
        if called.get("ok"):
            return {
                "ok": True,
                "source": "mcp",
                "connector": c.get("id"),
                "tool": tname,
                "text": called.get("text") or "",
            }
    return {"ok": False, "reason": "no_tool_matched"}


def register_mcp_routes(app=None):
    app = app or globals().get("app")
    if app is None:
        logger.warning("jagx_mcp: no app object")
        return

    from fastapi import Body

    @app.post("/mcp/list_tools")
    def list_tools_ep(body: dict = Body(...)):
        return mcp_list_tools((body.get("url") or "").strip(), body.get("token"))

    @app.post("/mcp/call")
    def call_tool_ep(body: dict = Body(...)):
        return mcp_call_tool(
            (body.get("url") or "").strip(),
            (body.get("name") or "").strip(),
            body.get("arguments") or {},
            body.get("token"),
        )

    @app.post("/mcp/run")
    def run_ep(body: dict = Body(...)):
        return run_connectors_for_message(
            (body.get("message") or "").strip(),
            body.get("connectors") or [],
        )

    @app.get("/mcp/info")
    def mcp_info():
        return {
            "builtin": list(BUILTIN.keys()),
            "endpoints": [
                "POST /mcp/list_tools",
                "POST /mcp/call",
                "POST /mcp/run",
                "GET /mcp/info",
            ],
            "note": "OAuth remote MCPs need a user token. Built-in tools are free.",
        }

    logger.info("JagX MCP routes registered: /mcp/list_tools /mcp/call /mcp/run /mcp/info")
