"""
JagX AI — Streaming multi-agent (SSE events).

Events:
  {"event":"status","message":"..."}
  {"event":"agent_start","agent":"engineer"}
  {"event":"agent_done","agent":"engineer","provider":"..."}
  {"event":"token","token":"..."}
  {"event":"attachment","attachment":{...}}
  {"event":"done","agents_used":[...],"provider":"multi-agent"}
  {"event":"error","detail":"..."}

Created by JagX & JRILICENSE
"""
from __future__ import annotations

import json
import logging
from typing import Iterator, List, Optional

from jagx_api.agents import run_multi_agent

logger = logging.getLogger("jagx-ai")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def stream_multi_agent_sse(
    user_message: str,
    history: Optional[List[dict]] = None,
    *,
    memory_context: Optional[str] = None,
    force_agents: Optional[List[str]] = None,
    max_agents: int = 3,
    reason: bool = False,
) -> Iterator[str]:
    yield _sse({"event": "status", "message": "Routing specialists…"})
    try:
        if force_agents:
            for a in force_agents:
                yield _sse({"event": "agent_start", "agent": a})
        else:
            yield _sse({"event": "status", "message": "Selecting agents…"})

        max_a = 4 if reason else max_agents
        result = run_multi_agent(
            user_message,
            history=history,
            memory_context=memory_context,
            force_agents=force_agents,
            max_agents=max_a,
        )

        for item in result.get("specialist_results") or []:
            yield _sse({
                "event": "agent_done",
                "agent": item.get("agent"),
                "provider": item.get("provider"),
            })

        text = result.get("response") or ""
        buf = ""
        for ch in text:
            buf += ch
            if ch.isspace() or ch in ".,!?;:\n":
                if buf:
                    yield _sse({"event": "token", "token": buf})
                    buf = ""
        if buf:
            yield _sse({"event": "token", "token": buf})

        if result.get("attachment"):
            att = result["attachment"]
            slim = {k: v for k, v in att.items() if k != "image_base64"}
            if att.get("image_base64"):
                slim["has_image_base64"] = True
                slim["image_base64"] = att["image_base64"]
            yield _sse({"event": "attachment", "attachment": slim})

        yield _sse({
            "event": "done",
            "agents_used": result.get("agents_used"),
            "provider": result.get("provider"),
            "version": result.get("version"),
            "parallel": result.get("parallel"),
            "datetime_utc": result.get("datetime_utc"),
        })
    except Exception as exc:
        logger.exception("stream_multi_agent failed: %s", exc)
        yield _sse({"event": "error", "detail": str(exc)[:500]})
        yield _sse({"event": "done", "agents_used": [], "provider": "error"})
