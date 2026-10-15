"""JagX MCP connector layer — realtime SSE + builtin free tools.

Remote MCP (JSON-RPC over HTTP/SSE) + local builtins (web, wiki, weather).
Created by JagX & JRILICENSE.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("jagx-ai")
UA = os.environ.get("JAGX_OSM_USER_AGENT", "JagXAI/8.1 (mcp-realtime)")

def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=1, backoff_factor=0.2, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP = _session()

def _ddg(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Give a search query."
    try:
        r = HTTP.get("https://api.duckduckgo.com/", params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}, timeout=10, headers={"User-Agent": UA})
        data = r.json() if r.status_code == 200 else {}
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:800])
        for item in (data.get("RelatedTopics") or [])[:5]:
            if isinstance(item, dict) and item.get("Text"):
                parts.append("- " + item["Text"][:200])
        if parts:
            return "Web search:\n" + "\n".join(parts)
        return f'No instant answer for "{q}". Try https://duckduckgo.com/?q={quote(q)}'
    except Exception as e:
        return f"Search error: {e}"

def _wiki(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Give a Wikipedia topic."
    try:
        r = HTTP.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(q.replace(" ", "_")), timeout=10, headers={"User-Agent": UA})
        if r.status_code != 200:
            return f'No Wikipedia page for "{q}".'
        data = r.json()
        return f"{data.get('title') or q} (Wikipedia)\n{(data.get('extract') or '')[:1200]}"
    except Exception as e:
        return f"Wikipedia error: {e}"

def _weather(place: str) -> str:
    place = (place or "Lagos").strip()[:80]
    try:
        geo = HTTP.get("https://nominatim.openstreetmap.org/search", params={"q": place, "format": "json", "limit": 1}, timeout=10, headers={"User-Agent": UA})
        rows = geo.json() if geo.status_code == 200 else []
        if not rows:
            return f'Could not locate "{place}".'
        lat, lon = rows[0]["lat"], rows[0]["lon"]
        label = rows[0].get("display_name") or place
        w = HTTP.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current_weather": "true", "timezone": "auto"}, timeout=10)
        if w.status_code != 200:
            return f"Weather API error for {label}."
        cw = (w.json() or {}).get("current_weather") or {}
        return f"Weather for {label}:\nTemperature: {cw.get('temperature')}°C\nWind: {cw.get('windspeed')} km/h\nTime: {cw.get('time')}"
    except Exception as e:
        return f"Weather unavailable: {e}"

def _maps(q: str) -> str:
    q = (q or "").strip()[:100]
    if not q:
        return "Give a place name."
    try:
        r = HTTP.get("https://nominatim.openstreetmap.org/search", params={"q": q, "format": "json", "limit": 3}, timeout=10, headers={"User-Agent": UA})
        rows = r.json() if r.status_code == 200 else []
        if not rows:
            return f'No map result for "{q}".'
        return "Maps:\n" + "\n".join(f"- {row.get('display_name')} (lat={row.get('lat')}, lon={row.get('lon')})" for row in rows)
    except Exception as e:
        return f"Maps unavailable: {e}"

BUILTIN = {
    "jagx_web": lambda a: _ddg(str(a.get("query") or "")),
    "web": lambda a: _ddg(str(a.get("query") or "")),
    "wikipedia": lambda a: _wiki(str(a.get("query") or a.get("topic") or "")),
    "jagx_weather": lambda a: _weather(str(a.get("place") or a.get("query") or "Lagos")),
    "jagx_maps": lambda a: _maps(str(a.get("query") or a.get("place") or "")),
    "jagx_news": lambda a: _ddg((str(a.get("topic") or a.get("query") or "world") + " news").strip()),
}

def _mcp_headers(token=None):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "User-Agent": UA}
    if token:
        h["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    return h

def mcp_rpc(url, method, params=None, token=None, timeout=20, req_id=None):
    payload = {"jsonrpc": "2.0", "id": req_id or str(uuid.uuid4())[:8], "method": method, "params": params or {}}
    r = HTTP.post(url, json=payload, headers=_mcp_headers(token), timeout=timeout)
    text = r.text or ""
    ctype = (r.headers.get("content-type") or "").lower()
    if "event-stream" in ctype or text.lstrip().startswith("event:"):
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

def mcp_list_tools(url, token=None):
    if not url:
        return {"ok": False, "error": "Missing MCP URL"}
    try:
        mcp_rpc(url, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "JagX-AI", "version": "8.1"}}, token=token, timeout=10)
    except Exception:
        pass
    try:
        res = mcp_rpc(url, "tools/list", {}, token=token, timeout=12)
        if "error" in res:
            return {"ok": False, "error": res["error"]}
        tools = (res.get("result") or {}).get("tools") or []
        return {"ok": True, "tools": tools, "count": len(tools)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def mcp_call_tool(url, name, arguments=None, token=None, timeout=40):
    if not url or not name:
        return {"ok": False, "error": "url and name required"}
    try:
        res = mcp_rpc(url, "tools/call", {"name": name, "arguments": arguments or {}}, token=token, timeout=timeout)
        if "error" in res:
            return {"ok": False, "error": res["error"]}
        result = res.get("result") or res
        content = result.get("content") if isinstance(result, dict) else None
        if isinstance(content, list):
            texts = [str(c.get("text") if isinstance(c, dict) else c) for c in content]
            return {"ok": True, "text": "\n".join(texts), "raw": result}
        return {"ok": True, "text": json.dumps(result, ensure_ascii=False)[:8000], "raw": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def call_builtin(name, arguments=None):
    fn = BUILTIN.get(name)
    if not fn:
        return {"ok": False, "error": f"unknown builtin: {name}"}
    try:
        return {"ok": True, "source": "builtin", "tool": name, "text": fn(arguments or {})}
    except Exception as e:
        return {"ok": False, "tool": name, "error": str(e)}

def _pick_builtin(message, enabled_ids=None):
    m = (message or "").lower()
    ids = set(enabled_ids or [])
    def has(*names):
        return (not ids) or any(n in ids for n in names)
    if has("jagx_news") and re.search(r"\b(news|headline|breaking)\b", m):
        topic = next((w for w in ("nigeria", "africa", "world", "tech") if w in m), "")
        return ("jagx_news", {"topic": topic})
    if has("jagx_weather") and re.search(r"\b(weather|temperature|forecast)\b", m):
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

def run_connectors_for_message(message, connectors=None):
    enabled = [c for c in (connectors or []) if c.get("enabled", True)]
    ids = [str(c.get("id") or "") for c in enabled]
    pick = _pick_builtin(message, ids if ids else None)
    if pick:
        return call_builtin(pick[0], pick[1])
    remotes = [c for c in enabled if (c.get("url") or "").strip()]
    if not remotes:
        return call_builtin("jagx_web", {"query": message})
    for c in remotes:
        url = (c.get("url") or "").strip()
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
            return {"ok": True, "source": "mcp", "connector": c.get("id"), "tool": tname, "text": called.get("text") or ""}
    return {"ok": False, "reason": "no_tool_matched"}

def stream_connectors(message, connectors=None):
    t0 = time.time()
    yield {"phase": "start", "message": message[:200], "ts": t0}
    enabled = [c for c in (connectors or []) if c.get("enabled", True)]
    ids = [str(c.get("id") or "") for c in enabled]
    pick = _pick_builtin(message, ids if ids else None)
    if pick:
        tool_id, args = pick
        yield {"phase": "builtin_selected", "tool": tool_id, "args": args}
        result = call_builtin(tool_id, args)
        yield {"phase": "result", "source": "builtin", "tool": tool_id, "ok": result.get("ok"), "text": result.get("text") or result.get("error"), "latency_ms": int((time.time() - t0) * 1000)}
        yield {"phase": "done", "ok": bool(result.get("ok"))}
        return
    remotes = [c for c in enabled if (c.get("url") or "").strip()]
    if not remotes:
        yield {"phase": "builtin_selected", "tool": "jagx_web", "args": {"query": message}}
        result = call_builtin("jagx_web", {"query": message})
        yield {"phase": "result", "source": "builtin", "tool": "jagx_web", "ok": result.get("ok"), "text": result.get("text") or result.get("error"), "latency_ms": int((time.time() - t0) * 1000)}
        yield {"phase": "done", "ok": bool(result.get("ok"))}
        return
    for c in remotes:
        url = (c.get("url") or "").strip()
        token = c.get("token") or c.get("auth_token")
        cid = c.get("id") or url[:40]
        yield {"phase": "mcp_connect", "connector": cid, "url": url[:80]}
        listed = mcp_list_tools(url, token=token)
        if not listed.get("ok"):
            yield {"phase": "mcp_error", "connector": cid, "error": listed.get("error")}
            continue
        tools = listed.get("tools") or []
        yield {"phase": "mcp_tools", "connector": cid, "count": len(tools), "names": [(t.get("name") if isinstance(t, dict) else str(t)) for t in tools[:12]]}
        if not tools:
            continue
        tname = tools[0].get("name") if isinstance(tools[0], dict) else str(tools[0])
        yield {"phase": "mcp_call", "connector": cid, "tool": tname}
        called = mcp_call_tool(url, tname, {"query": message}, token=token)
        yield {"phase": "result", "source": "mcp", "connector": cid, "tool": tname, "ok": called.get("ok"), "text": called.get("text") or str(called.get("error")), "latency_ms": int((time.time() - t0) * 1000)}
        if called.get("ok"):
            yield {"phase": "done", "ok": True}
            return
    yield {"phase": "done", "ok": False, "reason": "no_tool_matched"}

def stream_connectors_sse(message, connectors=None):
    for event in stream_connectors(message, connectors):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

def register_routes(app, require_key=None):
    from typing import List, Optional as Opt
    from fastapi import Header, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    class _McpList(BaseModel):
        url: str = ""
        token: Opt[str] = None
    class _McpCall(BaseModel):
        url: str = ""
        name: str = ""
        arguments: Opt[dict] = None
        token: Opt[str] = None
    class _McpRun(BaseModel):
        message: str = ""
        connectors: Opt[List[dict]] = None
        stream: bool = False
    def _key(x_api_key: Opt[str]):
        if require_key is not None:
            return require_key(x_api_key)
        from jagx_api import auth
        key = (x_api_key or "").strip()
        if not auth.is_valid_key(key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        ok, info = auth.check_rate_limit(key)
        if not ok:
            raise HTTPException(status_code=429, detail=info)
        return key
    @app.get("/mcp/info")
    def mcp_info():
        return {"builtin": list(BUILTIN.keys()), "endpoints": ["GET /mcp/info", "POST /mcp/list_tools", "POST /mcp/call", "POST /mcp/run", "POST /mcp/stream"], "realtime": True, "note": "Builtins free/instant. /mcp/stream = live SSE."}
    @app.post("/mcp/list_tools")
    def mcp_list(req: _McpList, x_api_key: Opt[str] = Header(None, alias="x-api-key")):
        _key(x_api_key)
        return mcp_list_tools((req.url or "").strip(), req.token)
    @app.post("/mcp/call")
    def mcp_call(req: _McpCall, x_api_key: Opt[str] = Header(None, alias="x-api-key")):
        _key(x_api_key)
        if req.name in BUILTIN and not (req.url or "").strip():
            return call_builtin(req.name, req.arguments or {})
        return mcp_call_tool((req.url or "").strip(), (req.name or "").strip(), req.arguments or {}, req.token)
    @app.post("/mcp/run")
    def mcp_run(req: _McpRun, x_api_key: Opt[str] = Header(None, alias="x-api-key")):
        _key(x_api_key)
        msg = (req.message or "").strip()
        if not msg:
            raise HTTPException(status_code=400, detail="message required")
        if req.stream:
            def gen():
                for event in stream_connectors_sse(msg, req.connectors):
                    yield event
            return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        return run_connectors_for_message(msg, req.connectors)
    @app.post("/mcp/stream")
    def mcp_stream(req: _McpRun, x_api_key: Opt[str] = Header(None, alias="x-api-key")):
        _key(x_api_key)
        msg = (req.message or "").strip()
        if not msg:
            raise HTTPException(status_code=400, detail="message required")
        def gen():
            for event in stream_connectors_sse(msg, req.connectors):
                yield event
        return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    logger.info("JagX MCP realtime routes registered")
