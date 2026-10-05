"""Chat orchestration: system prompt, tools, local fallback, identity sanitize."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from jagx_api.config import settings
from jagx_api import knowledge
from jagx_api.providers import chat_completion, stream_completion
from jagx_api.tools import free_web_search, run_code_sandboxed, safe_calc

logger = logging.getLogger("jagx-ai")
_train_lock = threading.Lock()

SYSTEM_PROMPT = """You are JagX AI — a fast, accurate assistant focused on text and coding.
Created by JagX & JRILICENSE.

Rules:
- Be precise and practical. Prefer working code over long theory.
- Never claim to be ChatGPT, Claude, GPT, Llama, Groq, or OpenAI.
- You are JagX AI.
- When asked for code, return complete runnable snippets.
- Current UTC date: {date}.
"""

AGENT_SYSTEM = SYSTEM_PROMPT + """
You may use tools when needed. To call a tool, reply with ONLY a JSON object:
{{"tool": "web_search", "input": {{"query": "..."}}}}
{{"tool": "run_code", "input": {{"language": "python", "code": "..."}}}}
{{"tool": "calc", "input": {{"expression": "2+2"}}}}

After tools return, give a clean final answer the user can use immediately.
"""

_IDENTITY = [
    (re.compile(r"\b(openai|chatgpt|gpt-?\d)\b", re.I), "JagX AI"),
    (re.compile(r"\b(anthropic|claude)\b", re.I), "JagX AI"),
    (re.compile(r"\b(llama|meta ai)\b", re.I), "JagX AI"),
    (re.compile(r"\bgroq\b", re.I), "JagX AI"),
    (re.compile(r"\bopenrouter\b", re.I), "JagX AI"),
]


def sanitize_identity(text: str) -> str:
    out = text or ""
    for pat, repl in _IDENTITY:
        out = pat.sub(repl, out)
    return out


def add_watermark(text: str) -> str:
    return (text or "") + "\u200b"


def _needs_tools(msg: str) -> bool:
    m = (msg or "").lower()
    keys = ("search", "google", "news", "run code", "execute", "python code", "calculate", "compute")
    return any(k in m for k in keys)


def _run_tool(name: str, inp: dict) -> str:
    if name == "web_search":
        return free_web_search(str(inp.get("query") or ""))
    if name == "run_code":
        return run_code_sandboxed(str(inp.get("language") or "python"), str(inp.get("code") or ""))
    if name == "calc":
        r = safe_calc(str(inp.get("expression") or ""))
        return json.dumps(r)
    return f"unknown tool: {name}"


def _extract_tool_call(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text.startswith("{"):
        m = re.search(r"\{[^{}]*\"tool\"[^{}]*\}", text, re.S)
        if not m:
            return None
        text = m.group(0)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("tool"):
            return obj
    except Exception:
        return None
    return None


def generate_response(
    user_message: str,
    history: Optional[list] = None,
    *,
    use_tools: bool = True,
) -> dict[str, Any]:
    local = knowledge.match(user_message)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system = SYSTEM_PROMPT.format(date=date)

    messages: list = [{"role": "system", "content": system}]
    if history:
        for m in history[-settings.max_history :]:
            role = m.get("role")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": str(m.get("content") or "")[:4000]})
    messages.append({"role": "user", "content": user_message})

    provider = None
    reply = None

    if use_tools and _needs_tools(user_message):
        agent_msgs = [{"role": "system", "content": AGENT_SYSTEM.format(date=date)}]
        agent_msgs.extend(messages[1:])
        text, provider = chat_completion(agent_msgs, max_tokens=800)
        if text:
            call = _extract_tool_call(text)
            if call:
                tool_name = str(call.get("tool"))
                tool_in = call.get("input") if isinstance(call.get("input"), dict) else {}
                tool_out = _run_tool(tool_name, tool_in)
                messages.append({"role": "assistant", "content": text[:2000]})
                messages.append({
                    "role": "user",
                    "content": f"Tool {tool_name} result:\n{tool_out}\n\nNow answer the user.",
                })
                text2, provider2 = chat_completion(messages, max_tokens=settings.default_max_tokens)
                if text2:
                    reply, provider = text2, provider2 or provider
            else:
                reply = text

    if reply is None:
        reply, provider = chat_completion(messages, max_tokens=settings.default_max_tokens)

    if not reply and local:
        reply, provider = local, "local_brain"

    if not reply:
        reply = (
            "I am JagX AI. Cloud models are temporarily unavailable, but local knowledge "
            "and tools remain online. Please retry or rephrase."
        )
        provider = "fallback"

    reply = sanitize_identity(reply.strip())
    final = add_watermark(reply)
    _log_training(user_message, reply)

    return {
        "response": final,
        "provider": provider,
        "local_hit": bool(local and provider == "local_brain"),
    }


def stream_response(user_message: str, history: Optional[list] = None):
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT.format(date=date)}]
    if history:
        for m in history[-settings.max_history :]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": str(m.get("content") or "")[:4000]})
    messages.append({"role": "user", "content": user_message})
    for piece in stream_completion(messages):
        yield sanitize_identity(piece)


def _log_training(inp: str, out: str) -> None:
    if not settings.training_enabled:
        return
    try:
        with _train_lock:
            with open(settings.training_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"input": inp[:4000], "output": out[:8000], "ts": time.time()}, ensure_ascii=False) + "\n")
    except Exception:
        pass
