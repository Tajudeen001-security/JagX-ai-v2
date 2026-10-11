"""Multi-provider LLM client with resilient ordered fallback and streaming."""
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
    retries = Retry(total=1, connect=1, read=0, backoff_factor=0.25,
                    status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["POST"]))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP = _session()

def provider_status() -> dict:
    return {"groq":bool(settings.groq_api_key), "nvidia":bool(settings.nvidia_api_key),
            "openrouter":bool(settings.openrouter_api_key), "huggingface":bool(settings.hf_token)}

def _providers():
    items=[]
    if settings.groq_api_key:
        items.append(("groq","https://api.groq.com/openai/v1/chat/completions",settings.groq_api_key,settings.groq_model,{}))
    if settings.nvidia_api_key:
        items.append(("nvidia","https://integrate.api.nvidia.com/v1/chat/completions",settings.nvidia_api_key,
                      getattr(settings,"nvidia_model","meta/llama-3.3-70b-instruct"),{}))
    if settings.openrouter_api_key:
        items.append(("openrouter","https://openrouter.ai/api/v1/chat/completions",settings.openrouter_api_key,
                      settings.openrouter_model,{"HTTP-Referer":"https://github.com/Tajudeen001-security/JagX-ai-v2","X-Title":"JagX AI"}))
    if settings.hf_token:
        items.append(("huggingface","https://router.huggingface.co/v1/chat/completions",settings.hf_token,settings.hf_model,{}))
    return items

def _headers(key: str, extra: dict) -> dict:
    return {"Authorization":f"Bearer {key}","Content-Type":"application/json",**extra}

def _parse_content(response) -> str:
    payload=response.json()
    choices=payload.get("choices") or []
    if not choices:
        raise ValueError("provider returned no choices")
    message=choices[0].get("message") or {}
    content=message.get("content")
    if isinstance(content,list):
        content="".join(str(part.get("text","")) for part in content if isinstance(part,dict))
    if not isinstance(content,str) or not content.strip():
        raise ValueError("provider returned empty content")
    return content.strip()

def chat_completion(messages: list, *, max_tokens: Optional[int]=None,
                    temperature: Optional[float]=None) -> tuple:
    max_tokens=max_tokens or settings.default_max_tokens
    temperature=settings.temperature if temperature is None else temperature
    for name,url,key,model,extra in _providers():
        try:
            r=HTTP.post(url,headers=_headers(key,extra),
                json={"model":model,"messages":messages,"max_tokens":max_tokens,"temperature":temperature},
                timeout=45)
            if r.status_code == 200:
                try:
                    return _parse_content(r),name
                except Exception as exc:
                    logger.warning("%s malformed success response: %s",name,exc)
            else:
                logger.warning("%s status %s: %s",name,r.status_code,(r.text or "")[:240])
        except Exception as exc:
            logger.warning("%s request failed: %s",name,exc)
    return None,None

def stream_completion(messages: list, *, max_tokens: Optional[int]=None) -> Iterator[str]:
    """Try every configured streaming provider; if all fail, use one-shot fallback."""
    max_tokens=max_tokens or settings.default_max_tokens
    for name,url,key,model,extra in _providers():
        response=None
        yielded=False
        try:
            response=HTTP.post(url,headers=_headers(key,extra),
                json={"model":model,"messages":messages,"max_tokens":max_tokens,
                      "temperature":settings.temperature,"stream":True},
                timeout=(10,70),stream=True)
            if response.status_code != 200:
                logger.warning("stream %s status %s: %s",name,response.status_code,(response.text or "")[:200])
                continue
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line,bytes):
                    line=line.decode("utf-8",errors="replace")
                if not line.startswith("data:"):
                    continue
                payload=line[5:].strip()
                if payload=="[DONE]":
                    break
                try:
                    chunk=json.loads(payload)
                    choices=chunk.get("choices") or []
                    if not choices:
                        continue
                    piece=((choices[0].get("delta") or {}).get("content"))
                    if isinstance(piece,str) and piece:
                        yielded=True
                        yield piece
                except (ValueError,TypeError,KeyError):
                    continue
            if yielded:
                return
            logger.warning("stream %s ended without text; trying next provider",name)
        except Exception as exc:
            logger.warning("stream %s failed: %s",name,exc)
        finally:
            if response is not None:
                try: response.close()
                except Exception: pass
    text,_=chat_completion(messages,max_tokens=max_tokens)
    if text:
        # Chunk on punctuation/whitespace to keep fallback responsive.
        buf=""
        for ch in text:
            buf+=ch
            if ch.isspace() or ch in ".,!?;:":
                yield buf
                buf=""
        if buf:
            yield buf
