"""Built-in tools: web search, code sandbox, calc."""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any

from jagx_api.config import settings
from jagx_api.providers import HTTP

logger = logging.getLogger("jagx-ai")

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def free_web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo instant-answer style search (no API key)."""
    q = (query or "").strip()[:200]
    if not q:
        return "empty query"
    try:
        r = HTTP.get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=12,
        )
        if r.status_code != 200:
            return f"search failed ({r.status_code})"
        data = r.json()
        parts: list[str] = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:800])
        for topic in (data.get("RelatedTopics") or [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"][:300])
            elif isinstance(topic, dict) and topic.get("Topics"):
                for t in topic["Topics"][:2]:
                    if t.get("Text"):
                        parts.append(t["Text"][:300])
        return "\n".join(parts)[:2500] if parts else "No concise results found."
    except Exception as e:
        logger.warning("search: %s", e)
        return f"search error: {e}"


def run_code_sandboxed(language: str, code: str) -> str:
    """Execute code via Piston public API."""
    lang = (language or "python").lower().strip()
    code = (code or "")[: settings.max_code_len]
    if not code.strip():
        return "empty code"
    lang_map = {"py": "python", "js": "javascript", "ts": "typescript", "sh": "bash"}
    lang = lang_map.get(lang, lang)
    try:
        r = HTTP.post(
            f"{settings.piston_url}/execute",
            json={"language": lang, "version": "*", "files": [{"content": code}]},
            timeout=20,
        )
        if r.status_code != 200:
            return f"sandbox error ({r.status_code}): {r.text[:300]}"
        data = r.json()
        run = data.get("run") or {}
        out = (run.get("stdout") or "") + (run.get("stderr") or "")
        return (out or "(no output)")[:3000]
    except Exception as e:
        return f"sandbox error: {e}"


def safe_calc(expression: str) -> dict[str, Any]:
    """Evaluate arithmetic expression safely (AST only)."""
    expr = (expression or "").strip()
    if not expr or len(expr) > 200:
        return {"ok": False, "error": "invalid expression"}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval(tree)
        if not isinstance(value, (int, float)):
            return {"ok": False, "error": "non-numeric result"}
        return {"ok": True, "value": value}
    except Exception as e:
        return {"ok": False, "error": str(e)}
