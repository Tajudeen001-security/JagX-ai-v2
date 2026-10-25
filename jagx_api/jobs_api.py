"""Durable jobs API — chat (multi-agent) and coding jobs.

V10: tier priority, reason/reflect modes, worker stats.
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from jagx_api import auth, conversation_store, job_store, memory_store
from jagx_api.config import settings

router = APIRouter()


class JobRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    conversation_id: Optional[str] = None
    training_consent: bool = False
    force_agents: Optional[List[str]] = None
    max_agents: int = Field(default=3, ge=1, le=5)
    mode: str = Field(default="agents", pattern="^(fast|agents|reason)$")
    reason: bool = False
    reflect: bool = False


class CodingJobRequest(BaseModel):
    task: str = Field(..., min_length=8, max_length=16000)
    conversation_id: Optional[str] = None


def _owner(key: str) -> str:
    return hashlib.sha256(("jagx-job-v1:" + key).encode()).hexdigest()


def _key(value: Optional[str]) -> str:
    key = (value or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok, info = auth.check_rate_limit(key)
    if not ok:
        raise HTTPException(status_code=429, detail=info)
    return key


@router.post("/jobs", status_code=202)
def create_job(
    req: JobRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """Enqueue a durable chat job (runs multi-agent in the background)."""
    key = _key(x_api_key)
    info = auth.key_info(key) or {}
    tier = str(info.get("tier") or "free")
    priority = job_store.priority_for_tier(tier)

    history = []
    if req.conversation_id:
        saved = conversation_store.get(key, req.conversation_id, limit=settings.max_history)
        if saved is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = [{"role": m["role"], "content": m["content"]} for m in saved["messages"]]
        conversation_store.append(key, req.conversation_id, "user", req.message)

    hits = memory_store.recall(
        hashlib.sha256(("jagx-memory-v1:" + key).encode()).hexdigest(),
        req.message,
    )
    memory_context = "\n".join(str(hit.get("text", "")) for hit in hits) if hits else None

    reason = bool(req.reason) or (req.mode == "reason")
    payload = {
        "message": req.message,
        "history": history,
        "memory_context": memory_context,
        "training_consent": req.training_consent,
        "conversation_id": req.conversation_id,
        "api_key": key,
        "force_agents": req.force_agents,
        "max_agents": req.max_agents,
        "mode": req.mode,
        "reason": reason,
        "reflect": bool(req.reflect),
        "tier": tier,
    }
    job = job_store.enqueue(_owner(key), "chat", payload, priority=priority)
    return {
        "ok": True,
        "job": job,
        "poll_url": f"/jobs/{job['id']}",
        "engine": "multi-agent",
        "priority": priority,
        "tier": tier,
        "note": "Background job uses JagX multi-agent. Paid tiers are claimed first. Poll until finished.",
    }


@router.get("/jobs/worker/status")
def worker_status(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    """Queue depth and worker health (authenticated)."""
    _key(x_api_key)
    stats = job_store.worker_stats()
    stats["worker"] = "jagx-job-worker"
    stats["version"] = "10.0.0"
    return {"ok": True, **stats}


@router.get("/jobs")
def list_jobs(limit: int = 20, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    return {"ok": True, "jobs": job_store.list_for(_owner(_key(x_api_key)), limit)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    job = job_store.get(_owner(_key(x_api_key)), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "job": job}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    job = job_store.request_cancel(_owner(_key(x_api_key)), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "job": job}


@router.post("/coding/jobs", status_code=202)
def create_coding_job(
    req: CodingJobRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    key = _key(x_api_key)
    info = auth.key_info(key) or {}
    tier = str(info.get("tier") or "free")
    priority = job_store.priority_for_tier(tier)

    history = []
    if req.conversation_id:
        saved = conversation_store.get(key, req.conversation_id, limit=settings.max_history)
        if saved is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = [{"role": m["role"], "content": m["content"]} for m in saved["messages"]]

    owner = _owner(key)
    hits = memory_store.recall(
        hashlib.sha256(("jagx-memory-v1:" + key).encode()).hexdigest(),
        req.task,
    )
    memory_context = "\n".join(str(hit.get("text", "")) for hit in hits) if hits else None
    payload = {
        "message": req.task,
        "history": history,
        "memory_context": memory_context,
        "conversation_id": req.conversation_id,
        "tier": tier,
    }
    job = job_store.enqueue(owner, "coding_plan", payload, priority=priority)
    return {
        "ok": True,
        "job": job,
        "poll_url": f"/jobs/{job['id']}",
        "engine": "multi-agent",
        "priority": priority,
        "tier": tier,
        "note": (
            "Task continues in the background via Engineer + Architect. "
            "Paid tiers are claimed first. Repository edits stay disabled until a trusted workspace is configured."
        ),
    }
