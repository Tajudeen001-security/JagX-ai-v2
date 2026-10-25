"""Background worker for durable AI jobs.

Chat jobs use JagX multi-agent (V9.9 parallel specialists + reason/reflect).
Coding jobs use coding_worker (Engineer + Architect).

Created by JagX & JRILICENSE
"""
from __future__ import annotations

import logging
import threading

from jagx_api import coding_worker, conversation_store, job_store
from jagx_api.agents import run_multi_agent
from jagx_api.chat import generate_response
from jagx_api.config import settings

logger = logging.getLogger("jagx-ai")
_stop = threading.Event()
_thread = None


def _run_chat_job(job: dict) -> None:
    jid = job["id"]
    payload = job["payload"] or {}
    message = payload.get("message") or ""
    history = payload.get("history")
    memory_context = payload.get("memory_context")
    conversation_id = payload.get("conversation_id")
    api_key = payload.get("api_key") or ""

    job_store.checkpoint(
        jid,
        {"step": "starting", "completed_steps": [], "message": "Preparing multi-agent task"},
    )
    if job_store.cancellation_requested(jid):
        return

    result = None
    try:
        job_store.checkpoint(
            jid,
            {
                "step": "multi_agent",
                "completed_steps": ["request"],
                "message": "Running specialists",
            },
        )
        result = run_multi_agent(
            message,
            history=history,
            memory_context=memory_context,
            force_agents=payload.get("force_agents"),
            max_agents=int(payload.get("max_agents") or 3),
            reason=bool(payload.get("reason") or payload.get("mode") == "reason"),
            reflect=bool(payload.get("reflect")),
        )
    except Exception as exc:
        logger.warning("Multi-agent job %s failed, falling back to chat: %s", jid, exc)
        job_store.checkpoint(
            jid,
            {
                "step": "fallback_chat",
                "completed_steps": ["request"],
                "message": "Multi-agent failed; using single-path chat",
            },
        )
        result = generate_response(
            message,
            history,
            memory_context=memory_context,
            training_consent=bool(payload.get("training_consent", False)),
        )

    if job_store.cancellation_requested(jid):
        return
    if not isinstance(result, dict):
        raise RuntimeError("AI engine returned an invalid result")

    if conversation_id and api_key and result.get("response"):
        try:
            conversation_store.append(
                api_key,
                conversation_id,
                "assistant",
                str(result.get("response") or ""),
                result.get("provider"),
            )
        except Exception as exc:
            logger.warning("Could not append job result to conversation: %s", exc)

    job_store.checkpoint(
        jid,
        {
            "step": "completed",
            "completed_steps": ["request", "response"],
            "message": "Task finished",
            "agents_used": result.get("agents_used"),
        },
    )
    job_store.finish(jid, result)


def _run_one(job: dict) -> None:
    if job_store.cancellation_requested(job["id"]):
        return

    if job["kind"] == "coding_plan":
        try:
            result = coding_worker.run(job)
            if job_store.cancellation_requested(job["id"]):
                return
            job_store.finish(job["id"], result)
        except Exception as exc:
            logger.exception("Coding job %s failed", job["id"])
            job_store.fail(
                job["id"],
                f"{type(exc).__name__}: {str(exc)[:1000]}",
                retry=True,
            )
        return

    if job["kind"] != "chat":
        job_store.fail(job["id"], "Unsupported job kind")
        return

    try:
        _run_chat_job(job)
    except Exception as exc:
        logger.exception("Background job %s failed", job["id"])
        job_store.fail(
            job["id"],
            f"{type(exc).__name__}: {str(exc)[:1000]}",
            retry=True,
        )


def _loop() -> None:
    while not _stop.is_set():
        try:
            job = job_store.claim_next()
            if job:
                _run_one(job)
                continue
        except Exception:
            logger.exception("Background job worker iteration failed")
        _stop.wait(settings.job_poll_seconds)


def start() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    job_store.init()
    _thread = threading.Thread(target=_loop, name="jagx-job-worker", daemon=True)
    _thread.start()
    logger.info("Durable job worker started (multi-agent chat); database=%s", settings.jobs_db)


def stop() -> None:
    _stop.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=3)
