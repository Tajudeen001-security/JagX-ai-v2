"""Multi-provider LLM client with ordered fallback."""

from __future__ import annotations

import json
import logging
from typing import Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from jagx_api.config import settings

logger = logging.getLogger("jagx-ai")


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


HTTP = _session()


def provider_status() -> dict:
    return {
        "groq": bool(settings.groq_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "huggingface": bool(settings.hf_token),
        "nvidia": bool(settings.nvidia_api_key),
    }


def chat_completion(
    messages: list,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> tuple:
    max_tokens = max_tokens or settings.default_max_tokens
    temperature = settings.temperature if temperature is None else temperature

    if settings.groq_api_key:
        try:
            r = HTTP.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.groq_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=28,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], "groq"
            logger.warning("Groq status %s: %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.warning("Groq error: %s", e)

    if settings.openrouter_api_key:
        try:
            r = HTTP.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Tajudeen001-security/JagX-ai-v2",
                    "X-Title": "JagX AI",
                },
                json={
                    "model": settings.openrouter_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=35,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], "openrouter"
            logger.warning("OpenRouter status %s", r.status_code)
        except Exception as e:
            logger.warning("OpenRouter error: %s", e)

    if settings.hf_token:
        try:
            r = HTTP.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.hf_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.hf_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=40,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], "huggingface"
            logger.warning("HF status %s", r.status_code)
        except Exception as e:
            logger.warning("HF error: %s", e)

    return None, None


def stream_completion(
    messages: list,
    *,
    max_tokens: Optional[int] = None,
) -> Iterator[str]:
    max_tokens = max_tokens or settings.default_max_tokens
    if settings.groq_api_key:
        try:
            r = HTTP.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.groq_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": settings.temperature,
                    "stream": True,
                },
                timeout=60,
                stream=True,
            )
            if r.status_code == 200:
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            yield piece
                    except Exception:
                        continue
                return
        except Exception as e:
            logger.warning("stream groq: %s", e)

    text, _ = chat_completion(messages, max_tokens=max_tokens)
    if text:
        yield text
