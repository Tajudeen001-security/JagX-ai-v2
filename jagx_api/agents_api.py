"""
JagX AI V9 — Agents API routes

New FastAPI endpoints for multi-agent execution.
Matches existing JagX-ai-v2 auth pattern (jobs_api style).

Created by JagX & JRILICENSE
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from jagx_api import auth
from jagx_api.agents import (
    list_capabilities,
    run_coding_agent,
    run_multi_agent,
    run_research_agent,
)

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


class AgentsRunResponse(BaseModel):
    response: str
    agents_used: List[str]
    specialist_results: List[Dict[str, Any]]
    provider: str
    version: str = "9.0.0"
    created_by: str = "JagX & JRILICENSE"


@router.get("/capabilities")
def agents_capabilities():
    """Public list of available agents and endpoints."""
    return list_capabilities()


@router.post("/run", response_model=AgentsRunResponse)
def agents_run(
    body: AgentsRunRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Full multi-agent run.
    Router selects specialists → specialists execute → synthesizer merges.
    """
    _key(x_api_key)
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    try:
        result = run_multi_agent(
            body.message,
            history=history,
            memory_context=body.memory_context,
            force_agents=body.force_agents,
            max_agents=body.max_agents,
        )
        return AgentsRunResponse(**result)
    except Exception as exc:
        logger.exception("agents/run failed: %s", exc)
        raise HTTPException(status_code=500, detail="Multi-agent run failed") from exc


@router.post("/coding", response_model=AgentsRunResponse)
def agents_coding(
    body: AgentsRunRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Fast path optimised for software engineering tasks.
    Forces Engineer + Architect.
    """
    _key(x_api_key)
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    try:
        result = run_coding_agent(
            body.message,
            history=history,
            memory_context=body.memory_context,
        )
        return AgentsRunResponse(**result)
    except Exception as exc:
        logger.exception("agents/coding failed: %s", exc)
        raise HTTPException(status_code=500, detail="Coding agent failed") from exc


@router.post("/research", response_model=AgentsRunResponse)
def agents_research(
    body: AgentsRunRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Fast path optimised for research and evidence-based answers.
    Forces Researcher + Africa Context.
    """
    _key(x_api_key)
    history = [{"role": t.role, "content": t.content} for t in (body.history or [])]
    try:
        result = run_research_agent(
            body.message,
            history=history,
            memory_context=body.memory_context,
        )
        return AgentsRunResponse(**result)
    except Exception as exc:
        logger.exception("agents/research failed: %s", exc)
        raise HTTPException(status_code=500, detail="Research agent failed") from exc
