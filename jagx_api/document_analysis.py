"""
JagX AI V9.5 — Document / file analysis.
Analyze text, code, or extracted PDF content with multi-agent + optional reflection.
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from jagx_api.agents import run_multi_agent
from jagx_api.artifacts import build_response_artifacts

logger = logging.getLogger("jagx-ai")

_MAX_CONTENT = 80000


def analyze_document(
    content: str,
    filename: str = "document.txt",
    question: str = "Summarize this document and extract key points.",
    mime_type: str = "text/plain",
) -> Dict[str, Any]:
    content = (content or "")[:_MAX_CONTENT]
    filename = (filename or "document.txt")[:200]
    question = (question or "Summarize this document and extract key points.")[:2000]

    lower_name = filename.lower()
    is_code = any(lower_name.endswith(ext) for ext in (
        ".py", ".js", ".ts", ".dart", ".java", ".go", ".rs", ".cpp", ".c", ".tsx", ".jsx", ".sql"
    ))
    is_json = lower_name.endswith(".json") or mime_type.endswith("json")

    doc_block = f"--- BEGIN DOCUMENT: {filename} ({mime_type}) ---\n{content}\n--- END DOCUMENT ---"
    user_msg = (
        f"{question}\n\n"
        f"Analyze the following document carefully.\n"
        f"{doc_block}"
    )

    force = ["engineer", "architect"] if is_code else (["researcher", "engineer"] if is_json else ["researcher", "africa"])
    result = run_multi_agent(
        user_msg,
        force_agents=force,
        max_agents=2,
        reason=True,
        reflect=True,
    )

    arts = build_response_artifacts(result.get("response") or "", result.get("attachment"))
    arts.insert(0, {
        "type": "document",
        "title": filename,
        "mime_type": mime_type,
        "chars": len(content),
        "preview": content[:400],
    })

    return {
        "response": result.get("response"),
        "agents_used": result.get("agents_used"),
        "specialist_results": result.get("specialist_results"),
        "provider": result.get("provider", "multi-agent"),
        "version": "9.5.0",
        "created_by": "JagX & JRILICENSE",
        "datetime_utc": result.get("datetime_utc"),
        "artifacts": arts,
        "filename": filename,
        "reason": True,
        "reflect": True,
        "thinking_summary": result.get("thinking_summary"),
    }
