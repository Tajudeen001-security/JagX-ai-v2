"""Optional local GGUF inference using llama-cpp-python. No model is downloaded automatically."""
from __future__ import annotations
import os
import threading
from pathlib import Path
from jagx_api.config import settings

_lock = threading.Lock()
_model = None
_model_path = ""

def status() -> dict:
    path = os.environ.get("JAGX_GGUF_MODEL", "").strip()
    installed = False
    try:
        import llama_cpp  # noqa: F401
        installed = True
    except Exception:
        pass
    exists = bool(path and Path(path).is_file())
    return {"enabled": bool(installed and exists), "runtime_installed": installed,
            "model_configured": bool(path), "model_file_found": exists,
            "model_path": path if path else None,
            "note": "Local GGUF inference requires llama-cpp-python and a compatible GGUF file. No automatic download."}

def generate(messages: list, max_tokens: int = 700) -> str | None:
    global _model, _model_path
    path = os.environ.get("JAGX_GGUF_MODEL", "").strip()
    if not path or not Path(path).is_file():
        return None
    try:
        from llama_cpp import Llama
    except Exception:
        return None
    with _lock:
        if _model is None or _model_path != path:
            _model = Llama(model_path=path, n_ctx=int(os.environ.get("JAGX_LOCAL_N_CTX","2048")),
                           n_threads=max(1,int(os.environ.get("JAGX_LOCAL_THREADS",str(os.cpu_count() or 2)))),
                           verbose=False)
            _model_path = path
        try:
            result = _model.create_chat_completion(messages=messages, max_tokens=max(64,min(max_tokens,2048)),
                                                   temperature=settings.temperature)
            content = result["choices"][0]["message"].get("content")
            return content.strip() if isinstance(content,str) and content.strip() else None
        except Exception:
            return None
