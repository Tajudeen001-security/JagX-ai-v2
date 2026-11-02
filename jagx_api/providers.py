"""Multi-provider LLM client with resilient ordered fallback and streaming."""
from __future__ import annotations
import json
import logging
from typing import Iterator, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from jagx_api.config import settings
from jagx_api import provider_resilience as resilience

logger = logging.getLogger("jagx-ai")


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=1, connect=1, read=0, backoff_factor=0.25,
                    status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["POST"]))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


HTTP = _session()


def provider_status() -> dict:
    return {"groq": bool(settings.groq_api_key), "nvidia": bool(settings.nvidia_api_key),
            "openrouter": bool(settings.openrouter_api_key), "huggingface": bool(settings.hf_token)}


def _providers():
    items = []
    if settings.groq_api_key:
        items.append(("groq", "https://api.groq.com/openai/v1/chat/completions", settings.groq_api_key, settings.groq_model, {}))
    if settings.nvidia_api_key:
        models = list(getattr(settings, "nvidia_models", []) or [])
        primary = getattr(settings, "nvidia_model", "meta/llama-3.3-70b-instruct")
        if primary not in models:
            models.insert(0, primary)
        # Attempt distinct enabled NIM models before falling through to another provider.
        for model in dict.fromkeys(models):
            items.append(("nvidia", "https://integrate.api.nvidia.com/v1/chat/completions", settings.nvidia_api_key, model, {}))
    if settings.openrouter_api_key:
        items.append(("openrouter", "https://openrouter.ai/api/v1/chat/completions", settings.openrouter_api_key,
                      settings.openrouter_model, {"HTTP-Referer": "https://github.com/Tajudeen001-security/JagX-ai-v2", "X-Title": "JagX AI"}))
    if settings.hf_token:
        items.append(("huggingface", "https://router.huggingface.co/v1/chat/completions", settings.hf_token, settings.hf_model, {}))
    return items


def _headers(key: str, extra: dict) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **extra}


def _parse_content(response) -> str:
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("provider returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("provider returned empty content")
    return content.strip()


def chat_completion(messages: list, *, max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None, preferred_provider: Optional[str] = None) -> tuple:
    max_tokens = max_tokens or settings.default_max_tokens
    temperature = settings.temperature if temperature is None else temperature
    if settings.offline_mode:
        return None, None
    providers = _providers()
    if preferred_provider:
        providers.sort(key=lambda item: 0 if item[0] == preferred_provider else 1)
    attempted_nvidia = set()
    for name, url, key, model, extra in providers:
        # Only check the circuit breaker once for the NVIDIA provider, so one
        # unavailable model does not prevent trying the remaining configured models.
        if name != "nvidia" or name not in attempted_nvidia:
            if not resilience.allow(name):
                logger.warning("%s circuit open; skipping provider", name)
                if name == "nvidia":
                    attempted_nvidia.add(name)
                continue
        if name == "nvidia":
            attempted_nvidia.add(name)
        try:
            r = HTTP.post(url, headers=_headers(key, extra),
                json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
                timeout=settings.llm_timeout_seconds)
            if r.status_code == 200:
                try:
                    parsed = _parse_content(r)
                    resilience.success(name)
                    return parsed, name
                except Exception as exc:
                    logger.warning("%s/%s malformed response: %s", name, model, type(exc).__name__)
                    resilience.failure(name, error=f"malformed response: {type(exc).__name__}")
            else:
                logger.warning("%s/%s status %s: %s", name, model, r.status_code, (r.text or "")[:180])
                # Count one provider failure per model attempt; later NVIDIA candidates still run.
                resilience.failure(name, error=f"HTTP {r.status_code}")
        except Exception as exc:
            logger.warning("%s/%s request failed: %s", name, model, type(exc).__name__)
            resilience.failure(name, error=type(exc).__name__)
    return None, None


def stream_completion(messages: list, *, max_tokens: Optional[int] = None) -> Iterator[str]:
    """Try configured providers/models for streaming; fall back to one-shot completion."""
    max_tokens = max_tokens or settings.default_max_tokens
    attempted_nvidia = set()
    for name, url, key, model, extra in _providers():
        if name != "nvidia" or name not in attempted_nvidia:
            if not resilience.allow(name):
                logger.warning("stream %s circuit open; skipping provider", name)
                if name == "nvidia":
                    attempted_nvidia.add(name)
                continue
        if name == "nvidia":
            attempted_nvidia.add(name)
        response = None
        yielded = False
        try:
            response = HTTP.post(url, headers=_headers(key, extra),
                json={"model": model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": settings.temperature, "stream": True},
                timeout=(min(10, settings.llm_timeout_seconds), settings.llm_timeout_seconds), stream=True)
            if response.status_code != 200:
                logger.warning("stream %s/%s status %s", name, model, response.status_code)
                resilience.failure(name, error=f"HTTP {response.status_code}")
                continue
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    piece = (choices[0].get("delta") or {}).get("content")
                    if isinstance(piece, str) and piece:
                        yielded = True
                        yield piece
                except (ValueError, TypeError, KeyError):
                    continue
            if yielded:
                resilience.success(name)
                return
            logger.warning("stream %s/%s ended without text", name, model)
            resilience.failure(name, error="empty stream")
        except Exception as exc:
            logger.warning("stream %s/%s failed: %s", name, model, type(exc).__name__)
            resilience.failure(name, error=type(exc).__name__)
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
    text, _ = chat_completion(messages, max_tokens=max_tokens)
    if text:
        buf = ""
        for ch in text:
            buf += ch
            if ch.isspace() or ch in ".,!?;:":
                yield buf
                buf = ""
        if buf:
            yield buf


def provider_health() -> dict:
    """Network reachability and breaker state; does not spend tokens."""
    from jagx_api.provider_resilience import snapshot
    breaker = snapshot()
    targets = {
        "groq": ("https://api.groq.com", bool(settings.groq_api_key)),
        "nvidia": ("https://integrate.api.nvidia.com", bool(settings.nvidia_api_key)),
        "openrouter": ("https://openrouter.ai", bool(settings.openrouter_api_key)),
        "huggingface": ("https://router.huggingface.co", bool(settings.hf_token)),
    }
    result = {}
    for name, (url, configured) in targets.items():
        status = breaker.get(name, {"circuit": "closed", "failures": 0, "retry_after_seconds": 0})
        if not configured:
            result[name] = {**status, "configured": False, "reachable": None, "detail": "not configured"}
            continue
        if status["circuit"] == "open":
            result[name] = {**status, "configured": True, "reachable": None, "detail": "temporarily skipped by circuit breaker"}
            continue
        try:
            response = HTTP.get(url, timeout=(2, 3), allow_redirects=True)
            result[name] = {**status, "configured": True, "reachable": True, "http_status": response.status_code,
                            "detail": "network reachable; key, quota and model not validated"}
        except Exception as exc:
            result[name] = {**status, "configured": True, "reachable": False, "detail": type(exc).__name__}
    return result
