"""JagX v13 — OpenAI-compatible LLM client with native tool calling and streaming.

**Local-first.** The provider chain in ``config.py`` puts Ollama, llama.cpp and
vLLM ahead of every hosted endpoint, so a machine with a local runtime never
needs a third-party API key. All of them speak the same ``/chat/completions``
protocol, so this one client covers all of them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import httpx

from .config import ProviderConfig, settings

logger = logging.getLogger("jagx.llm")


class LLMError(RuntimeError):
    """Raised when no provider could serve the request."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    finish_reason: str = ""


def _resolve_provider(preferred: str = "") -> ProviderConfig:
    providers = settings.providers()
    if not providers:
        raise LLMError(
            "No LLM provider is reachable. JagX is local-first: start a local runtime "
            "(e.g. `ollama serve` with `ollama pull qwen2.5:7b`) and it will be picked up "
            "automatically — no API key needed."
        )
    if preferred:
        for p in providers:
            if p.name == preferred or p.model == preferred:
                return p
    return providers[0]


def _headers(provider: ProviderConfig) -> dict:
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    if provider.name == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/Tajudeen001-security/JagX-ai-v2"
        headers["X-Title"] = "JagX AI"
    return headers


def _payload(
    provider: ProviderConfig,
    messages: list[dict],
    *,
    tools: Optional[list[dict]],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict:
    body: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def _parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": str(raw)[:2000]}


def complete(
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: str = "",
) -> LLMResult:
    """One non-streaming turn. Returns text and/or tool calls."""
    provider = _resolve_provider(model)
    body = _payload(
        provider,
        messages,
        tools=tools,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.max_tokens,
        stream=False,
    )
    url = f"{provider.base_url.rstrip('/')}/chat/completions"
    try:
        with httpx.Client(timeout=settings.request_timeout) as client:
            resp = client.post(url, headers=_headers(provider), json=body)
    except httpx.HTTPError as exc:
        raise LLMError(f"{provider.name} request failed: {type(exc).__name__}") from exc

    if resp.status_code >= 400:
        raise LLMError(f"{provider.name} returned HTTP {resp.status_code}: {resp.text[:400]}")

    try:
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
    except (ValueError, IndexError, AttributeError) as exc:
        raise LLMError(f"{provider.name} returned a malformed response") from exc

    calls: list[ToolCall] = []
    for i, raw in enumerate(message.get("tool_calls") or []):
        fn = raw.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        calls.append(
            ToolCall(
                id=raw.get("id") or f"call_{i}",
                name=name,
                arguments=_parse_arguments(fn.get("arguments")),
            )
        )

    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))

    return LLMResult(
        content=content.strip(),
        tool_calls=calls,
        provider=provider.name,
        model=provider.model,
        finish_reason=choice.get("finish_reason") or "",
    )


def stream_turn(
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: str = "",
) -> Iterator[dict]:
    """Stream one agent turn, accumulating tool calls as they arrive.

    Yields:
      {"type": "token", "text": "..."}
      {"type": "end", "content": str, "tool_calls": [ToolCall], "provider": str, "model": str}
    """
    provider = _resolve_provider(model)
    body = _payload(
        provider,
        messages,
        tools=tools,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.max_tokens,
        stream=True,
    )
    url = f"{provider.base_url.rstrip('/')}/chat/completions"

    content_parts: list[str] = []
    pending: dict[int, dict] = {}

    try:
        with httpx.Client(timeout=settings.request_timeout) as client:
            with client.stream("POST", url, headers=_headers(provider), json=body) as resp:
                if resp.status_code >= 400:
                    text = resp.read().decode("utf-8", "replace")[:400]
                    raise LLMError(f"{provider.name} returned HTTP {resp.status_code}: {text}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        content_parts.append(piece)
                        yield {"type": "token", "text": piece}

                    for raw in delta.get("tool_calls") or []:
                        idx = raw.get("index", 0)
                        slot = pending.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if raw.get("id"):
                            slot["id"] = raw["id"]
                        fn = raw.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
    except httpx.HTTPError as exc:
        raise LLMError(f"{provider.name} stream failed: {type(exc).__name__}") from exc

    calls = [
        ToolCall(
            id=slot["id"] or f"call_{idx}",
            name=slot["name"],
            arguments=_parse_arguments(slot["arguments"]),
        )
        for idx, slot in sorted(pending.items())
        if slot["name"]
    ]

    yield {
        "type": "end",
        "content": "".join(content_parts).strip(),
        "tool_calls": calls,
        "provider": provider.name,
        "model": provider.model,
    }


def health() -> dict:
    providers = settings.providers()
    local = [p for p in providers if p.local]
    return {
        "configured": bool(providers),
        "local_first": bool(local),
        "providers": [
            {"name": p.name, "model": p.model, "local": p.local} for p in providers
        ],
        "active": providers[0].name if providers else None,
        "note": (
            "Running on a local open-weight runtime — no third-party API key in use."
            if local and providers and providers[0].local
            else "No local runtime detected; a hosted endpoint is configured."
            if providers
            else "No provider reachable. Start Ollama (or vLLM / llama.cpp) — no key required."
        ),
    }
