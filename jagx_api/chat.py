"""Chat orchestration: system prompt, tools, local fallback, identity sanitize."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from jagx_api.config import settings
from jagx_api import knowledge
from jagx_api.providers import chat_completion, stream_completion
from jagx_api.tools import run_tool

logger = logging.getLogger("jagx-ai")
_train_lock = threading.Lock()

SYSTEM_PROMPT = """You are JagX AI, created by JagX & JRILICENSE: an ambitious, capable, warm, clear, and resourceful African-built AI assistant.

Voice and quality:
- Sound natural, confident, curious, and human-readable—not robotic, repetitive, stiff, or needlessly formal.
- Answer the actual question first. Use plain language, useful structure, examples, and a little warmth where appropriate.
- Multilingual behavior: detect the language the user is actually using and reply in that language by default. Support code-switching naturally, including Nigerian English and common Nigerian Pidgin, and do not force English when the user writes another language. Preserve names, technical terms, and scripts accurately. If the language or meaning is uncertain, ask a short clarification instead of pretending to understand. For speech transcripts, tolerate punctuation errors and likely transcription mistakes, but confirm critical numbers, names, coordinates, and commands before consequential actions.
- Match the user's language and level of detail. Explain unfamiliar technical steps patiently, especially for mobile-first builders.
- Be honest about uncertainty, live access, and limitations. Never claim you tested, connected, remembered, browsed, trained, or changed something unless that actually happened.
- Think carefully, check assumptions, compare options, and give practical next steps. Do not pad answers or repeat the question.
- For code, provide complete runnable snippets where practical, explain required environment variables, and never expose secrets.
- Do not claim to be another AI brand. Your identity is JagX AI.
- Protect each user's private data. Only use memories belonging to the authenticated user; do not infer sensitive traits or create secret profiles. Respect requests to view, correct, or delete saved memories.
- For real-world products, engineering, health, legal, financial, vehicle, aviation, drone, robotics, and industrial questions, state assumptions, distinguish estimates from verified facts, and encourage qualified validation where mistakes could cause harm. For vehicles and aircraft, provide decision support, simulations, checklists, anomaly explanations, and recommendations—not direct control commands. Require independent safety interlocks, sensor validation, fail-safe behavior, and an authorised human/operator for safety-critical decisions. Never present untested generated code as flight-ready or road-ready.
- Support lawful, defensive, humanitarian, and safety-focused public-sector or military-adjacent work (such as logistics, maintenance, disaster response, accessibility, and non-targeting image analysis). Do not help plan violence, select/track people for attacks, build autonomous weapons, or bypass human authorization for high-impact decisions.
- Current UTC date: {date}.
"""

AGENT_SYSTEM = SYSTEM_PROMPT + """
You may use tools when needed. To call a tool, reply with ONLY a JSON object (no markdown):
{{"tool": "web_search", "input": {{"query": "..."}}}}
{{"tool": "run_code", "input": {{"language": "python", "code": "..."}}}}
{{"tool": "calc", "input": {{"expression": "2+2"}}}}
{{"tool": "generate_pdf", "input": {{"title": "Report", "sections": [{{"heading": "Intro", "content": "..."}}]}}}}
{{"tool": "generate_cv", "input": {{"name": "Ada Lovelace", "title": "Engineer", "summary": "...", "experience": [{{"role": "...", "company": "...", "dates": "...", "bullets": ["..."]}}], "skills": ["Python"]}}}}
{{"tool": "generate_portfolio", "input": {{"name": "Name", "title": "Title", "about": "...", "projects": [{{"name": "...", "description": "...", "link": "https://..."}}], "skills": [], "contact": {{"email": ""}}}}}}
{{"tool": "generate_image", "input": {{"prompt": "a red robot in a forest"}}}}

After tools return, give a clean final answer the user can use immediately.
"""

_IDENTITY = [
    (re.compile(r"\b(openai|chatgpt|gpt-?\d)\b", re.I), "JagX AI"),
    (re.compile(r"\b(anthropic|claude)\b", re.I), "JagX AI"),
    (re.compile(r"\b(llama|meta ai)\b", re.I), "JagX AI"),
    (re.compile(r"\bgroq\b", re.I), "JagX AI"),
    (re.compile(r"\bopenrouter\b", re.I), "JagX AI"),
]

_TOOL_HINTS = (
    "search", "google", "news", "run code", "execute", "python code",
    "calculate", "compute", "pdf", "document", "cv", "resume",
    "curriculum", "portfolio", "generate image", "draw", "picture",
)


def sanitize_identity(text: str) -> str:
    out = text or ""
    for pat, repl in _IDENTITY:
        out = pat.sub(repl, out)
    return out


def add_watermark(text: str) -> str:
    return (text or "") + "\u200b"


def _needs_tools(msg: str) -> bool:
    m = (msg or "").lower()
    return any(k in m for k in _TOOL_HINTS)


def _extract_tool_call(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if not text.startswith("{"):
        m = re.search(
            r"\{\s*\"tool\"\s*:\s*\"[^\"]+\"\s*,\s*\"input\"\s*:\s*\{.*?\}\s*\}",
            text,
            re.S,
        )
        if not m:
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
    attachment = None

    if use_tools and _needs_tools(user_message):
        agent_msgs = [{"role": "system", "content": AGENT_SYSTEM.format(date=date)}]
        agent_msgs.extend(messages[1:])
        text, provider = chat_completion(agent_msgs, max_tokens=900)
        if text:
            call = _extract_tool_call(text)
            if call:
                tool_name = str(call.get("tool"))
                tool_in = call.get("input") if isinstance(call.get("input"), dict) else {}
                tool_out, attachment = run_tool(tool_name, tool_in)
                messages.append({"role": "assistant", "content": text[:2000]})
                messages.append({
                    "role": "user",
                    "content": f"Tool {tool_name} result:\n{tool_out}\n\nNow answer the user briefly.",
                })
                text2, provider2 = chat_completion(messages, max_tokens=settings.default_max_tokens)
                if text2:
                    reply, provider = text2, provider2 or provider
                else:
                    reply = tool_out
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

    out: dict[str, Any] = {
        "response": final,
        "provider": provider,
        "local_hit": bool(local and provider == "local_brain"),
    }
    if attachment:
        out["attachment"] = attachment
    return out


def stream_response(user_message: str, history: Optional[list] = None) -> Iterator[str]:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT.format(date=date)}]
    if history:
        for m in history[-settings.max_history :]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": str(m.get("content") or "")[:4000]})
    messages.append({"role": "user", "content": user_message})

    local = knowledge.match(user_message)
    if local and len(user_message.strip()) < 80:
        yield sanitize_identity(local)
        return

    had = False
    for piece in stream_completion(messages):
        clean = sanitize_identity(piece)
        if clean:
            had = True
            yield clean
    if not had:
        yield sanitize_identity(
            "JagX AI is online, but no streamed tokens were returned. Retry or disable stream."
        )


def stream_sse(user_message: str, history: Optional[list] = None) -> Iterator[str]:
    try:
        for piece in stream_response(user_message, history):
            yield f"data: {json.dumps({'token': piece}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        logger.warning("stream_sse error: %s", e)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"


def _log_training(inp: str, out: str) -> None:
    if not settings.training_enabled:
        return
    try:
        with _train_lock:
            with open(settings.training_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"input": inp[:4000], "output": out[:8000], "ts": time.time()}, ensure_ascii=False) + "\n")
    except Exception:
        pass
