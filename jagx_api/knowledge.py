"""Local knowledge brain — works offline without cloud LLMs."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("jagx-ai")

_KB: list = []


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s\+\-\*\?\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_brain(root: Optional[str] = None) -> int:
    global _KB
    root_path = Path(root or ".")
    entries: list = []

    knowledge = root_path / "jagx_knowledge.json"
    if knowledge.exists():
        try:
            data = json.loads(knowledge.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for row in data:
                    q = row.get("question") or row.get("q") or ""
                    a = row.get("answer") or row.get("a") or ""
                    if q and a:
                        entries.append({"keys": [_normalize(q)], "answer": a})
        except Exception as e:
            logger.warning("load jagx_knowledge: %s", e)

    for path in sorted(root_path.glob("knowledge_pack*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for row in data:
                keys = row.get("k") or row.get("keys") or []
                a = row.get("a") or row.get("answer") or ""
                if isinstance(keys, str):
                    keys = [keys]
                keys_n = [_normalize(k) for k in keys if k]
                if keys_n and a:
                    entries.append({"keys": keys_n, "answer": a})
        except Exception as e:
            logger.warning("load %s: %s", path.name, e)

    _KB = entries
    logger.info("Local brain loaded: %d entries", len(_KB))
    return len(_KB)


def match(user_message: str) -> Optional[str]:
    if not _KB:
        return None
    q = _normalize(user_message)
    if not q or len(q) > 240:
        return None
    best = None
    best_score = 0.0
    q_tokens = set(q.split())
    for entry in _KB:
        for k in entry["keys"]:
            if not k:
                continue
            if q == k:
                return entry["answer"]
            if k in q or q in k:
                score = len(k) / max(len(q), 1)
                if score > best_score:
                    best_score = score
                    best = entry["answer"]
            k_tokens = set(k.split())
            if not k_tokens:
                continue
            overlap = len(q_tokens & k_tokens) / len(k_tokens)
            if overlap >= 0.7 and overlap > best_score:
                best_score = overlap
                best = entry["answer"]
    if best_score >= 0.55:
        return best
    return None


def size() -> int:
    return len(_KB)
