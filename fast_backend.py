"""Low-latency production launcher for JagX AI.

Keeps the existing API implementation intact while selecting the fastest
available inference path and using Render's assigned PORT correctly.
"""
from __future__ import annotations

import os
from typing import Optional

import requests
import uvicorn

import app as jagx

jagx.GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
jagx.OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
_FAST_TIMEOUT = float(os.getenv("JAGX_LLM_TIMEOUT", "8"))


def _post(url: str, *, headers: dict, payload: dict, timeout: float) -> Optional[str]:
    try:
        response = jagx.HTTP.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            return None
        choices = response.json().get("choices") or []
        return choices[0].get("message", {}).get("content") if choices else None
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def fast_call_llm(messages: list, max_tokens: int = 700) -> Optional[str]:
    """Use the low-latency provider first and short-circuit successful calls."""
    max_tokens = max(1, min(int(max_tokens), int(os.getenv("JAGX_MAX_OUTPUT_TOKENS", "1200"))))

    if jagx.GROQ_API_KEY:
        result = _post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {jagx.GROQ_API_KEY}", "Content-Type": "application/json"},
            payload={"model": jagx.GROQ_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.4},
            timeout=_FAST_TIMEOUT,
        )
        if result:
            return result

    if jagx.OPENROUTER_API_KEY:
        result = _post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {jagx.OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            payload={"model": jagx.OPENROUTER_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.4},
            timeout=_FAST_TIMEOUT,
        )
        if result:
            return result

    if jagx.HF_TOKEN:
        return _post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {jagx.HF_TOKEN}", "Content-Type": "application/json"},
            payload={"model": os.getenv("HF_MODEL", "Qwen/Qwen3-8B"), "messages": messages, "max_tokens": max_tokens, "temperature": 0.4},
            timeout=_FAST_TIMEOUT,
        )
    return None


jagx.call_llm = fast_call_llm

# Cheap health endpoint for Render; it does not invoke an LLM.
@jagx.app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "service": "jagx-ai-v2"}


if __name__ == "__main__":
    uvicorn.run(
        jagx.app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
