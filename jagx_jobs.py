"""
JagX background jobs — Grok-style "keep working when you leave".
Jobs live on the server (JSON file + worker thread), not only on the phone.
Created by JagX & JRILICENSE. Free-tier friendly (small user counts).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("jagx-ai")

JOBS_FILE = os.environ.get("JAGX_JOBS_FILE", "jagx_jobs.json")
MAX_JOBS = int(os.environ.get("JAGX_MAX_JOBS", "200"))
WORKER_SLEEP = int(os.environ.get("JAGX_JOB_WORKER_SECONDS", "8"))

_lock = threading.Lock()
_worker_started = False

try:
    HTTP  # type: ignore  # noqa: F821
except NameError:
    HTTP = requests.Session()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> List[dict]:
    if not os.path.exists(JOBS_FILE):
        return []
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(jobs: List[dict]) -> None:
    jobs = jobs[:MAX_JOBS]
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)


def create_job(goal: str, user_id: str = "guest", meta: Optional[dict] = None) -> dict:
    goal = (goal or "").strip()
    if not goal:
        return {"ok": False, "error": "goal required"}
    job = {
        "id": str(uuid.uuid4()),
        "goal": goal[:2000],
        "user_id": (user_id or "guest")[:120],
        "status": "queued",  # queued | running | done | failed
        "logs": [f"Queued at {_now()}"],
        "result": None,
        "created_at": _now(),
        "updated_at": _now(),
        "meta": meta or {},
    }
    with _lock:
        jobs = _load()
        jobs.insert(0, job)
        _save(jobs)
    logger.info("job created %s for %s", job["id"], job["user_id"])
    return {"ok": True, "job": job}


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        for j in _load():
            if j.get("id") == job_id:
                return j
    return None


def list_jobs(user_id: str = "", limit: int = 30) -> List[dict]:
    with _lock:
        jobs = _load()
    if user_id:
        jobs = [j for j in jobs if j.get("user_id") == user_id]
    return jobs[: max(1, min(limit, 100))]


def _append_log(job: dict, line: str) -> None:
    logs = job.setdefault("logs", [])
    logs.append(line)
    if len(logs) > 80:
        job["logs"] = logs[-80:]
    job["updated_at"] = _now()


def _run_one(job: dict) -> dict:
    """Execute a single job on the server (no phone required)."""
    goal = job.get("goal") or ""
    _append_log(job, "Running on server…")

    # 1) Optional free tools
    tool_text = ""
    try:
        import jagx_mcp as jm

        tool_res = jm.run_connectors_for_message(
            goal,
            [{"id": "jagx_web", "enabled": True}, {"id": "jagx_news", "enabled": True}],
        )
        if tool_res.get("ok") and tool_res.get("text"):
            tool_text = str(tool_res["text"])[:3000]
            _append_log(job, f"Tool: {tool_res.get('tool') or tool_res.get('source')}")
    except Exception as e:
        _append_log(job, f"Tools skipped: {e}")

    prompt = goal
    if tool_text:
        prompt = f"{goal}\n\n[TOOL RESULT]\n{tool_text}"

    # 2) Use existing generate_response if core loaded it
    reply = None
    try:
        gen = globals().get("generate_response")
        if callable(gen):
            _append_log(job, "Model generating…")
            out = gen(prompt, history=[])
            if isinstance(out, dict):
                reply = out.get("response") or out.get("message") or str(out)
            else:
                reply = str(out)
    except Exception as e:
        _append_log(job, f"generate_response: {e}")

    # 3) Fallback: call own /chat if available
    if not reply:
        try:
            port = os.environ.get("PORT", "10000")
            r = HTTP.post(
                f"http://127.0.0.1:{port}/chat",
                json={"message": prompt},
                headers={"Content-Type": "application/json"},
                timeout=90,
            )
            if r.status_code == 200:
                data = r.json()
                reply = data.get("response") or data.get("message")
                _append_log(job, "Answer via /chat")
        except Exception as e:
            _append_log(job, f"/chat fallback: {e}")

    if not reply and tool_text:
        reply = tool_text
        _append_log(job, "Returned tool result only")

    if reply:
        job["status"] = "done"
        job["result"] = str(reply)[:12000]
        _append_log(job, "Done")
    else:
        job["status"] = "failed"
        job["result"] = "Could not complete job. Try again when the API is warm."
        _append_log(job, "Failed")
    job["updated_at"] = _now()
    return job


def _worker_loop() -> None:
    logger.info("JagX job worker started")
    while True:
        try:
            job = None
            with _lock:
                jobs = _load()
                for j in jobs:
                    if j.get("status") == "queued":
                        j["status"] = "running"
                        j["updated_at"] = _now()
                        _append_log(j, "Picked by worker")
                        _save(jobs)
                        job = j
                        break
            if job:
                try:
                    done = _run_one(dict(job))
                    with _lock:
                        jobs = _load()
                        for i, j in enumerate(jobs):
                            if j.get("id") == done.get("id"):
                                jobs[i] = done
                                break
                        _save(jobs)
                except Exception as e:
                    logger.warning("job run error: %s", e)
                    with _lock:
                        jobs = _load()
                        for j in jobs:
                            if j.get("id") == job.get("id"):
                                j["status"] = "failed"
                                j["result"] = str(e)[:500]
                                _append_log(j, f"Error: {e}")
                                j["updated_at"] = _now()
                        _save(jobs)
            else:
                time.sleep(WORKER_SLEEP)
        except Exception as e:
            logger.warning("job worker loop: %s", e)
            time.sleep(15)


def start_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    if os.environ.get("JAGX_JOBS_ENABLED", "true").lower() != "true":
        return
    _worker_started = True
    t = threading.Thread(target=_worker_loop, name="jagx-jobs", daemon=True)
    t.start()


def register_job_routes(app=None) -> None:
    app = app or globals().get("app")
    if app is None:
        logger.warning("jagx_jobs: no app")
        return

    from fastapi import Body, Query

    @app.post("/jobs")
    def jobs_create(body: dict = Body(...)):
        return create_job(
            goal=str(body.get("goal") or ""),
            user_id=str(body.get("user_id") or "guest"),
            meta=body.get("meta") if isinstance(body.get("meta"), dict) else {},
        )

    @app.get("/jobs")
    def jobs_list(user_id: str = Query(""), limit: int = Query(30)):
        return {"jobs": list_jobs(user_id=user_id, limit=limit)}

    @app.get("/jobs/{job_id}")
    def jobs_get(job_id: str):
        j = get_job(job_id)
        if not j:
            return {"ok": False, "error": "not found"}
        return {"ok": True, "job": j}

    @app.get("/jobs_info")
    def jobs_info():
        with _lock:
            n = len(_load())
        return {
            "file": JOBS_FILE,
            "count": n,
            "worker": _worker_started,
            "hint": "POST /jobs {goal, user_id} then GET /jobs/{id} — works after app close",
        }

    start_worker()
    logger.info("JagX job routes registered: POST /jobs GET /jobs GET /jobs/{id}")
