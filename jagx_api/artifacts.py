"""
JagX AI V9.5 — Structured artifacts for canvas-style UIs.
Richer extraction: code, tables, links, attachments.
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
        if first.startswith("#") and ("/" in first or ".py" in first or ".dart" in first or ".ts" in first or ".js" in first):
            title = first.lstrip("#").strip()[:80]
        artifacts.append({
            "type": "code",
            "title": title,
            "language": lang,
            "content": body,
            "lines": body.count("\n") + 1,
        })
    return artifacts


def extract_link_artifacts(text: str) -> List[Dict[str, Any]]:
    arts: List[Dict[str, Any]] = []
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)]+)\)", text or ""):
        arts.append({"type": "link", "title": m.group(1)[:120], "url": m.group(2)})
    for m in re.finditer(r"(?<![(\[])(https?://[^\s)>\]]+)", text or ""):
        url = m.group(1).rstrip(".,;:")
        if not any(a.get("url") == url for a in arts):
            arts.append({"type": "link", "title": url[:60], "url": url})
    return arts[:20]


def extract_table_artifacts(text: str) -> List[Dict[str, Any]]:
    arts: List[Dict[str, Any]] = []
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", lines[i + 1] or ""):
            table_lines = [lines[i]]
            j = i + 1
            while j < len(lines) and "|" in lines[j]:
                table_lines.append(lines[j])
                j += 1
            if len(table_lines) >= 2:
                arts.append({
                    "type": "table",
                    "title": f"table_{len(arts) + 1}",
                    "content": "\n".join(table_lines),
                    "rows": len(table_lines) - 1,
                })
            i = j
        else:
            i += 1
    return arts


def normalize_attachment(att: Optional[dict]) -> Optional[Dict[str, Any]]:
    if not att or not isinstance(att, dict):
        return None
    t = att.get("type") or "file"
    out: Dict[str, Any] = {"type": t, "title": att.get("filename") or att.get("title") or t}
    for k in ("url", "content_base64", "mime_type", "filename", "prompt", "provider", "width", "height", "size_bytes"):
        if att.get(k) is not None:
            out[k] = att[k]
    if att.get("image_base64") and "content_base64" not in out:
        out["content_base64"] = att["image_base64"]
        out["type"] = "image"
    return out


def build_response_artifacts(
    response_text: str,
    attachment: Optional[dict] = None,
    extra: Optional[List[dict]] = None,
) -> List[Dict[str, Any]]:
    arts = extract_code_artifacts(response_text)
    arts.extend(extract_table_artifacts(response_text))
    arts.extend(extract_link_artifacts(response_text))
    norm = normalize_attachment(attachment)
    if norm:
        arts.append(norm)
    if extra:
        for e in extra:
            if isinstance(e, dict) and e.get("type"):
                arts.append(e)
    return arts
