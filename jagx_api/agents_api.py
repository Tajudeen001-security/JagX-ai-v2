"""
JagX AI V9.4 — Agents API routes
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from jagx_api import auth
from jagx_api.agent_stream import stream_multi_agent_sse
from jagx_api.agents import (
    list_capabilities,
    run_coding_agent,
    run_media_agent,
    run_multi_agent,
    run_research_agent,
)
from jagx_api.artifacts import build_response_artifacts
from jagx_api.vision import describe_image

logger = logging.getLogger("jagx-ai")
router = APIRouter(prefix="/agents", tags=["agents"])


def _key(value: Optional[str]) -> str:
    key = (value or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok, info = auth.check_rate_limit(key)
    if not ok:
        raise HTTPException(status_code=429, detail=info)
    return key


class HistoryTurn(BaseModel):
    role: str
    content: str


class AgentsRunRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12000)
    history: Optional[List[HistoryTurn]] = None
    memory_context: Optional[str] = None
    force_agents: Optional[List[str]] = None
    max_agents: int = Field(default=3, ge=1, le=5)
    stream: bool = False
    reason: bool = False


class VisionRequest(BaseModel):
    image_base64: str = Field(..., min_length=32)
    prompt: str = Field(default="Describe this image in detail.", max_length=2000)
    mime_type: str = "image/jpeg"


class AgentsRunResponse(BaseModel):
    response: str
    agents_used: List[str]
    specialist_results: List[Dict[str, Any]]
    provider: str
    version: str = "9.4.0"
    created_by: str = "JagX & JRILICENSE"
    datetime_utc: Optional[str] = None
    attachment: Optional[Dict[str, Any]] = None
    artifacts: Optional[List[Dict[str, Any]]] = None
    parallel: Optional[bool] = None
    reason: Optional[bool] = None
    thinking_summary: Optional[str] = None


def _pack(result: dict) -> AgentsRunResponse:
    arts = build_response_artifacts(result.get("response") or "", result.get("attachment"))
    data = {k: v for k, v in result.items() if k in AgentsRunResponse.model_fields}
    data["artifacts"] = arts or None
    data["version"] = "9.4.0"
    return AgentsRunResponse(**data)


@router.get("/capabilities")
def agents_capabilities():
    caps = list_capabilities()
    caps["version"] = "9.4.0"
    extra = [
        "streaming_multi_agent",
        "vision_input",
        "artifacts",
        "extended_reasoning",
        "tool_loop_engineer",
        "code_interpreter",
        "chat_mode_agents",
    ]
    caps["capabilities"] = list(dict.fromkeys((caps.get("capabilities") or []) + extra))
    caps["endpoints"] = list(dict.fromkeys((caps.get("endpoints") or []) + [
        "POST /agents/stream", "POST /agents/vision"
    ]))
    return caps


@router.post("/run")
def agents_run(body: AgentsRunRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    _key(x_api_key)
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    if body.stream:
        return StreamingResponse(
            stream_multi_agent_sse(
                body.message,
                history=history,
                memory_context=body.memory_context,
                force_agents=body.force_agents,
                max_agents=body.max_agents,
                reason=body.reason,
            ),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    try:
        result = run_multi_agent(
            body.message,
            history=history,
            memory_context=body.memory_context,
            force_agents=body.force_agents,
            max_agents=4 if body.reason else body.max_agents,
            reason=body.reason,
        )
        return _pack(result)
    except Exception as exc:
        logger.exception("agents/run failed: %s", exc)
        raise HTTPException(status_code=500, detail="Multi-agent run failed") from exp


@router.post("/stream")
def agents_stream(body: AgentsRunRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    _key(x_api_key)
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    return StreamingResponse(
        stream_multi_agent_sse(
            body.message,
            history=history,
            memory_context=body.memory_context,
            force_agents=body.force_agents,
            max_agents=body.max_agents,
            reason=body.reason,
        ),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/coding", response_model=AgentsRunResponse)
def agents_coding(body: AgentsRunRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    _key(x_api_key)
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    try:
        return _pack(run_coding_agent(
            body.message, history=history, memory_context=body.memory_context, reason=body.reason
        ))
    except Exception as exc:
        logger.exception("agents/coding failed: %s", exp)
        raise HTTPException(status_code=500, detail="Coding agent failed") from exp


@router.post("/research", response_model=AgentsRunResponse)
def agents_research(body: AgentsRunRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    _key(x_api_key)
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    try:
        return _pack(run_research_agent(
            body.message, history=history, memory_context=body.memory_context, reason=body.reason
        ))
    except Exception as exc:
        logger.exception("agents/research failed: %s", exp)
        raise HTTPException(status_code=500, detail="Research agent failed") from exp


@router.post("/media", response_model=AgentsRunResponse)
def agents_media(body: AgentsRunRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    _key(x_api_key)
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    try:
        return _pack(run_media_agent(body.message, history=history, memory_context=body.memory_context))
    except Exception as exp:
        logger.exception("agents/media failed: %s", exp)
        raise HTTPException(status_code=500, detail="Media agent failed") from exp


@router.post("/vision")
def agents_vision(body: VisionRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    _key(x_api_key)
    try:
        return describe_image(body.image_base64, body.prompt, mime_type=body.mime_type)
    except ValueError as exp:
        raise HTTPException(status_code=400, detail=str(exc)) from exp
    except Exception as exp:
        logger.exception("agents/vision failed: %s", exp)
        raise HTTPException(status_code=500, detail="Vision failed") from exp
