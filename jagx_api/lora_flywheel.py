"""LoRA / SFT data flywheel — consented training export + quality filters.

Never trains on secrets, keys, or non-consented private memory.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from jagx_api.config import settings

logger = logging.getLogger("jagx-ai")

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|password|token|bearer\s+[a-z0-9._-]{12,}|sk-[a-z0-9]{10,}|ghp_[a-z0-9]{20,})"
)
_MIN_INPUT = 8
_MIN_OUTPUT = 20
_MAX_INPUT = 4000
_MAX_OUTPUT = 8000


def _path() -> Path:
    return Path(settings.training_file)


def stats() -> dict[str, Any]:
    p = _path()
    if not p.exists():
        return {
            "enabled": settings.training_enabled,
            "path": str(p),
            "rows": 0,
            "bytes": 0,
            "ready_for_lora": False,
        }
    rows = 0
    good = 0
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                try:
                    obj = json.loads(line)
                    if _is_quality(obj):
                        good += 1
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("training stats failed: %s", exc)
        return {"enabled": settings.training_enabled, "path": str(p), "error": str(exc)}
    return {
        "enabled": settings.training_enabled,
        "path": str(p),
        "rows": rows,
        "quality_rows": good,
        "bytes": p.stat().st_size,
        "ready_for_lora": good >= 50,
        "min_quality_rows_hint": 50,
    }


def _is_quality(obj: dict) -> bool:
    inp = str(obj.get("input") or obj.get("prompt") or "").strip()
    out = str(obj.get("output") or obj.get("response") or "").strip()
    if len(inp) < _MIN_INPUT or len(out) < _MIN_OUTPUT:
        return False
    if len(inp) > _MAX_INPUT or len(out) > _MAX_OUTPUT:
        return False
    if _SECRET_RE.search(inp) or _SECRET_RE.search(out):
        return False
    low = out.lower()
    if low.count("jagx") > 6 and len(out) < 200:
        return False
    return True


def iter_sft_rows(*, quality_only: bool = True, limit: int = 10000) -> Iterator[dict]:
    """Yield ChatML-style rows for LoRA SFT."""
    p = _path()
    if not p.exists():
        return
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if quality_only and not _is_quality(obj):
                continue
            inp = str(obj.get("input") or obj.get("prompt") or "").strip()[:_MAX_INPUT]
            out = str(obj.get("output") or obj.get("response") or "").strip()[:_MAX_OUTPUT]
            if not inp or not out:
                continue
            yield {
                "messages": [
                    {"role": "user", "content": inp},
                    {"role": "assistant", "content": out},
                ],
                "meta": {
                    "ts": obj.get("ts"),
                    "source": "jagx_consented",
                },
            }
            n += 1


def export_jsonl(dest: str | Path, *, quality_only: bool = True, limit: int = 10000) -> dict:
    dest = Path(dest)
    count = 0
    with dest.open("w", encoding="utf-8") as out:
        for row in iter_sft_rows(quality_only=quality_only, limit=limit):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return {"ok": True, "rows": count, "path": str(dest), "quality_only": quality_only}


def append_consented(inp: str, out: str, *, extra: Optional[dict] = None) -> bool:
    """Append one consented training row (caller must already check consent + settings)."""
    if not settings.training_enabled:
        return False
    row = {
        "input": (inp or "")[:_MAX_INPUT],
        "output": (out or "")[:_MAX_OUTPUT],
        "ts": time.time(),
    }
    if extra:
        row.update({k: v for k, v in extra.items() if k not in row})
    if not _is_quality(row):
        return False
    try:
        with _path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:
        logger.warning("training append failed: %s", exc)
        return False
