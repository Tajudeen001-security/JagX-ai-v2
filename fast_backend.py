"""Low-latency production launcher for JagX AI.

Keeps the existing API implementation intact while selecting the fastest
available inference path and using Render's assigned PORT correctly.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests
import uvicorn

import app as jagx

# Prefer a low-latency production model. Override with GROQ_MODEL when needed.
jagx.GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
jagx.OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

_FAST_TIMEOUT = float(os.getenv("JAGX_LLM_TIMEOUT", "8"))


def _post(url: str, *, headers: dict, payload: dict, timeout: float) -> Optional[str]:
    try:
        response = jagx.HTTP.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            return None
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message", {}).get("content")
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def fast_call_llm(messages: list, max_tokens: int = 700) -> Optional[str]:
    """Fast provider chain: Groq first, then explicitly configured fallbacks."""
    # Keep outputs short by default; callers can still request a larger budget.
    max_tokens = max(1, min(int(max_tokens), int(os.getenv("JAGX_MAX_OUTPUT_TOKENS", "1200"))))

    if jagx.GROQ_API_KEY:
        result = _post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {jagx.GROQ_API_KEY}", "Content-Type": "application/json"},
            payload={
                "model": jagx.GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.4,
                "stream": False,
            },
            timeout=_FAST_TIMEOUT,
        )
        if result:
            return result

    if jagx.OPENROUTER_API_KEY:
        result = _post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {jagx.OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            payload={
                "model": jagx.OPENROUTER_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.4,
            },
            timeout=_FAST_TIMEOUT,
        )
        if result:
            return result

    if jagx.HF_TOKEN:
        return _post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {jagx.HF_TOKEN}", "Content-Type": "application/json"},
            payload={
                "model": os.getenv("HF_MODEL", "Qwen/Qwen3-8B"),
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.4,
            },
            timeout=_FAST_TIMEOUT,
        )

    return None


# app.generate_response/run_agent resolve call_llm from the app module namespace.
jagx.call_llm = fast_call_llm


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(jagx.app, host="0.0.0.0", port=port, log_level=os.getenv("LOG_LEVEL", "info"))
