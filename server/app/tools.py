"""JagX v13 — the tool layer.

Every tool returns a plain string (the observation the model sees next turn).
Tools are small, side-effect-light and timeout-bounded.

**No tool requires a third-party API key.** Web search falls back to keyless
DuckDuckGo; code runs in the local sandbox; memory is local SQLite; the whole
multimodal surface runs on this machine.
"""

from __future__ import annotations

import ast
import json
import logging
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import quote

import httpx

from . import db, embeddings, media_tools, sandbox
from .config import settings

logger = logging.getLogger("jagx.tools")

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_UA = "Mozilla/5.0 (compatible; JagXAI/13.0; +https://github.com/Tajudeen001-security/JagX-ai-v2)"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]
    needs_user: bool = False
    category: str = "general"


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------
def _web_search(query: str, max_results: int = 5) -> str:
    if not settings.enable_web_tools:
        return "Web tools are disabled by the server operator (JAGX_ENABLE_WEB_TOOLS=false)."

    query = (query or "").strip()[:300]
    if not query:
        return "Error: empty query."
    n = max(1, min(int(max_results or 5), 8))

    # 1) Tavily — best quality when a key happens to be present (optional).
    if settings.tavily_api_key:
        try:
            with httpx.Client(timeout=settings.tool_timeout) as client:
                r = client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": settings.tavily_api_key, "query": query,
                          "max_results": n, "include_answer": True},
                )
            if r.status_code == 200:
                data = r.json()
                parts = []
                if data.get("answer"):
                    parts.append(f"Answer: {data['answer']}")
                for item in data.get("results") or []:
                    parts.append(
                        f"- {item.get('title')}\n  {item.get('url')}\n  {(item.get('content') or '')[:400]}"
                    )
                if parts:
                    return "\n".join(parts)[: settings.max_tool_output_chars]
        except httpx.HTTPError as exc:
            logger.warning("tavily failed: %s", exc)

    # 2) Serper (optional).
    if settings.serper_api_key:
        try:
            with httpx.Client(timeout=settings.tool_timeout) as client:
                r = client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": n},
                )
            if r.status_code == 200:
                data = r.json()
                parts = []
                if (data.get("answerBox") or {}).get("answer"):
                    parts.append(f"Answer: {data['answerBox']['answer']}")
                for item in (data.get("organic") or [])[:n]:
                    parts.append(
                        f"- {item.get('title')}\n  {item.get('link')}\n  {(item.get('snippet') or '')[:400]}"
                    )
                if parts:
                    return "\n".join(parts)[: settings.max_tool_output_chars]
        except httpx.HTTPError as exc:
            logger.warning("serper failed: %s", exc)

    # 3) Keyless DuckDuckGo — the default, zero-config path.
    try:
        with httpx.Client(timeout=settings.tool_timeout, follow_redirects=True) as client:
            r = client.get(
                f"https://html.duckduckgo.com/html/?q={quote(query)}",
                headers={"User-Agent": _UA},
            )
        if r.status_code == 200:
            pattern = re.compile(
                r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'class="result__snippet"[^>]*>(.*?)</(?:a|td)>',
                re.DOTALL,
            )
            out = []
            for href, title, snippet in pattern.findall(r.text)[:n]:
                title = re.sub(r"<[^>]+>", "", title).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                if title and href.startswith("http"):
                    out.append(f"- {title}\n  {href}\n  {snippet[:400]}")
            if out:
                return "\n".join(out)[: settings.max_tool_output_chars]
    except httpx.HTTPError as exc:
        logger.warning("ddg failed: %s", exc)

    return (
        "Search returned no results. The keyless search backend may be rate-limited; "
        "answer from your own knowledge and say the search was unavailable."
    )


def _fetch_url(url: str, max_chars: int = 6000) -> str:
    if not settings.enable_web_tools:
        return "Web tools are disabled by the server operator."
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://"
    try:
        with httpx.Client(
            timeout=settings.tool_timeout, follow_redirects=True, headers={"User-Agent": _UA}
        ) as client:
            r = client.get(url)
        if r.status_code >= 400:
            return f"Error: HTTP {r.status_code} fetching {url}"
        ctype = r.headers.get("content-type", "")
        text = r.text
        if "html" in ctype:
            text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'"), ("&quot;", '"')):
                text = text.replace(a, b)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        limit = max(500, min(int(max_chars or 6000), 20000))
        return text[:limit] or "(page had no extractable text)"
    except httpx.HTTPError as exc:
        return f"Error fetching {url}: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def _run_code(language: str, code: str) -> str:
    return sandbox.run_code(language, code)


def _calc(expression: str) -> str:
    expr = (expression or "").strip()
    if not expr or len(expr) > 300:
        return "Error: invalid expression."

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        value = _eval(ast.parse(expr, mode="eval"))
        return f"{expr} = {value}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def _remember(user_id: str, content: str, tags: str = "", layer: str = "semantic") -> str:
    content = (content or "").strip()
    if len(content) < 3:
        return "Error: nothing to remember."
    layer = layer if layer in ("semantic", "episodic", "procedural") else "semantic"
    db.add_memory(
        user_id, content, layer=layer, tags=tags, vector=embeddings.embed(content)
    )
    return f"Saved to {layer} memory: {content[:200]}"


def _recall(user_id: str, query: str) -> str:
    from . import memory

    hits = memory.retrieve(user_id, query or "", limit=6)
    if not hits:
        return "No relevant memories found."
    return "\n".join(f"- [{h['layer']}] {h['content']}" for h in hits)[: settings.max_tool_output_chars]


def _list_memories(user_id: str) -> str:
    rows = db.list_memories(user_id, limit=40)
    if not rows:
        return "No memories stored yet."
    return "\n".join(f"- [{m['layer']}] {m['content']}" for m in rows)[: settings.max_tool_output_chars]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
TOOLS: dict[str, Tool] = {}


def _register(tool: Tool) -> None:
    TOOLS[tool.name] = tool


_register(Tool(
    name="web_search",
    description=(
        "Search the live web. Use for current events, prices, docs, or anything that may have "
        "changed. Returns titles, URLs and snippets. Works with no API key."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {"type": "integer", "description": "How many results (1-8).", "default": 5},
        },
        "required": ["query"],
    },
    handler=_web_search,
    category="web",
))

_register(Tool(
    name="fetch_url",
    description="Fetch a web page and return its readable text. Use after web_search to read a source in full.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL."},
            "max_chars": {"type": "integer", "description": "Max characters to return.", "default": 6000},
        },
        "required": ["url"],
    },
    handler=_fetch_url,
    category="web",
))

_register(Tool(
    name="run_code",
    description=(
        "Execute code in a local sandbox and return stdout/stderr. Use for data analysis, maths, "
        "string processing, or verifying that code actually runs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "language": {"type": "string", "description": "python, javascript, typescript or bash.", "default": "python"},
            "code": {"type": "string", "description": "The full program to run."},
        },
        "required": ["language", "code"],
    },
    handler=_run_code,
    category="compute",
))

_register(Tool(
    name="calculator",
    description="Evaluate an arithmetic expression exactly. Use instead of doing mental maths.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. (1250*0.07)+89"}},
        "required": ["expression"],
    },
    handler=_calc,
    category="compute",
))

_register(Tool(
    name="remember",
    description=(
        "Save a durable fact about the user (preferences, project details, decisions) to long-term "
        "memory. Use when the user shares something worth recalling in future conversations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact, written as a standalone sentence."},
            "tags": {"type": "string", "description": "Optional comma-separated tags."},
            "layer": {"type": "string", "description": "semantic (a fact) or procedural (how to do something).", "default": "semantic"},
        },
        "required": ["content"],
    },
    handler=_remember,
    needs_user=True,
    category="memory",
))

_register(Tool(
    name="recall",
    description="Search long-term memory for facts about this user relevant to the current task.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look for."}},
        "required": ["query"],
    },
    handler=_recall,
    needs_user=True,
    category="memory",
))

_register(Tool(
    name="list_memories",
    description="List everything stored in long-term memory for this user.",
    parameters={"type": "object", "properties": {}},
    handler=_list_memories,
    needs_user=True,
    category="memory",
))

# --- multimodal surface (local, free, no API key) --------------------------
media_tools.register(_register, Tool)


def schemas() -> list[dict]:
    """OpenAI-format tool definitions."""
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in TOOLS.values()
    ]


def describe() -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters, "category": t.category}
        for t in TOOLS.values()
    ]


def execute(name: str, arguments: dict, *, user_id: str) -> tuple[str, bool]:
    """Run a tool. Returns (observation, ok)."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS)}", False

    args = dict(arguments or {})
    if tool.needs_user:
        args["user_id"] = user_id

    allowed = set(tool.parameters.get("properties", {})) | ({"user_id"} if tool.needs_user else set())
    args = {k: v for k, v in args.items() if k in allowed}

    try:
        return str(tool.handler(**args)), True
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}", False
    except Exception as exc:  # noqa: BLE001 - a tool failure must not kill the run
        logger.exception("tool %s failed", name)
        return f"Error: {name} failed: {type(exc).__name__}: {exc}", False
