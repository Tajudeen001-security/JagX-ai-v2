"""
JagX AI — Structured artifacts for canvas-style UIs.
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def extract_code_artifacts(text: str) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.S)
    for i, m in enumerate(pattern.finditer(text or "")):
        lang = (m.group(1) or "text").strip()
        body = (m.group(2) or "").strip()
        if len(body) < 8:
            continue
        title = f"code_{i + 1}.{lang}"
        first = body.splitlines()[0] if body else ""
        if first.startswith("#") and ("/" in first or ".py" in first or ".dart" in first):
            title = first.lstrip("#").strip()[:80]
        artifacts.append({"type": "code", "title": title, "language": lang, "content": body})
    return artifacts


def normalize_attachment(att: Optional[dict]) -> Optional[Dict[str, Any]]:
    if not att or not isinstance(att, dict):
        return None
    t = att.get("type") or "file"
    out: Dict[str, Any] = {"type": t, "title": att.get("filename") or att.get("title") or t}
    for k in ("url", "content_base64", "mime_type", "filename", "prompt", "provider", "width", "height"):
        if att.get(k) is not None:
            out[k] = att[k]
    if att.get("image_base64") and "content_base64" not in out:
        out["content_base64"] = att["image_base64"]
        out["type"] = "image"
    return out


def build_response_artifacts(response_text: str, attachment: Optional[dict] = None) -> List[Dict[str, Any]]:
    arts = extract_code_artifacts(response_text)
    norm = normalize_attachment(attachment)
    if norm:
        arts.append(norm)
    return arts
