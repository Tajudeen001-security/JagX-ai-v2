"""
JagX Bot jobs — Grok-style agent that keeps working after app close.
Multi-step: plan → tools → answer → optional GitHub file change → self-learn.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("jagx-ai")

JOBS_FILE = os.environ.get("JAGX_JOBS_FILE", "jagx_jobs.json")
MAX_JOBS = int(os.environ.get("JAGX_MAX_JOBS", "200"))
WORKER_SLEEP = int(os.environ.get("JAGX_JOB_WORKER_SECONDS", "8"))
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
DEFAULT_REPO = os.environ.get("JAGX_BOT_DEFAULT_REPO", "Tajudeen001-security/JagX-ai-v2")

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
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs[:MAX_JOBS], f, ensure_ascii=False, indent=2)


def create_job(goal: str, user_id: str = "guest", meta: Optional[dict] = None) -> dict:
    goal = (goal or "").strip()
    if not goal:
        return {"ok": False, "error": "goal required"}
    job = {
        "id": str(uuid.uuid4()),
        "goal": goal[:4000],
        "user_id": (user_id or "guest")[:120],
        "status": "queued",
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
    if len(logs) > 100:
        job["logs"] = logs[-100:]
    job["updated_at"] = _now()


def _chat(prompt: str, system: str = "") -> Optional[str]:
    """Talk to local model stack."""
    try:
        gen = globals().get("generate_response")
        if callable(gen):
            msg = f"{system}\n\n{prompt}" if system else prompt
            out = gen(msg, history=[])
            if isinstance(out, dict):
                return str(out.get("response") or out.get("message") or out)
            return str(out)
    except Exception as e:
        logger.info("gen: %s", e)
    try:
        port = os.environ.get("PORT", "10000")
        body = {"message": prompt}
        if system:
            body["message"] = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
        r = HTTP.post(
            f"http://127.0.0.1:{port}/chat",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("response") or data.get("message")
    except Exception as e:
        logger.info("/chat: %s", e)
    return None


def _tools(goal: str) -> str:
    try:
        import jagx_mcp as jm

        res = jm.run_connectors_for_message(
            goal,
            [
                {"id": "jagx_web", "enabled": True},
                {"id": "jagx_news", "enabled": True},
                {"id": "wikipedia", "enabled": True},
                {"id": "jagx_maps", "enabled": True},
            ],
        )
        if res.get("ok") and res.get("text"):
            return str(res["text"])[:4000]
    except Exception as e:
        logger.info("tools: %s", e)
    return ""


def _github_put(path: str, content: str, message: str, repo: str = "") -> str:
    """Write/update a file on GitHub when GITHUB_TOKEN is set (like editing your work)."""
    token = GITHUB_TOKEN.strip()
    if not token:
        return "GitHub write skipped: set GITHUB_TOKEN env on Render to allow bot file changes."
    repo = (repo or DEFAULT_REPO).strip()
    if "/" not in repo:
        return f"Invalid repo: {repo}"
    api = f"https://api.github.com/repos/{repo}/contents/{path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "JagX-Bot",
    }
    sha = None
    try:
        gr = HTTP.get(api, headers=headers, timeout=20)
        if gr.status_code == 200:
            sha = gr.json().get("sha")
    except Exception:
        pass
    body = {
        "message": message[:200],
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": "main",
    }
    if sha:
        body["sha"] = sha
    try:
        r = HTTP.put(api, headers=headers, json=body, timeout=30)
        if r.status_code in (200, 201):
            return f"GitHub updated: {repo}/{path}"
        return f"GitHub write failed ({r.status_code}): {r.text[:300]}"
    except Exception as e:
        return f"GitHub error: {e}"


def _maybe_github_action(goal: str, plan: str, answer: str, job: dict) -> str:
    """If goal asks to change code/files and token exists, attempt a safe write."""
    g = goal.lower()
    wants = any(
        w in g
        for w in (
            "push to github",
            "commit",
            "update the repo",
            "change the file",
            "edit app.py",
            "write to github",
            "fix the code",
            "deploy fix",
        )
    )
    if not wants:
        return ""
    _append_log(job, "GitHub change requested…")
    # Extract a simple fenced code block if the model produced one
    m = re.search(r"```(?:\w+)?\n([\s\S]{20,8000}?)```", answer)
    if not m:
        return "No code block in answer to commit. Ask bot to output a full file in a code fence."
    code = m.group(1)
    # Path hint
    path = "docs/BOT_OUTPUT.md"
    for cand in ("app.py", "jagx_mcp.py", "jagx_jobs.py", "README.md"):
        if cand in g:
            path = cand
            break
    if path == "docs/BOT_OUTPUT.md":
        code = f"# Bot output\n\nGoal: {goal}\n\n## Plan\n{plan}\n\n## Result\n\n{answer}\n"
    return _github_put(path, code, f"JagX Bot: {goal[:80]}")


def _self_learn_snippet(goal: str, result: str) -> None:
    try:
        path = globals().get("TRAINING_DATA_FILE", "jagx_training_data.jsonl")
        row = json.dumps(
            {"input": goal[:500], "output": (result or "")[:2000], "ts": _now()},
            ensure_ascii=False,
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(row + "\n")
        # Trigger idle self-learn if available
        try:
            import jagx_extensions as jx

            jx.run_self_learn_once()
        except Exception:
            pass
    except Exception as e:
        logger.info("self_learn: %s", e)


def _run_one(job: dict) -> dict:
    goal = job.get("goal") or ""
    _append_log(job, "JagX Bot engaged on server")

    # Step 1 — plan (Atlas-style)
    _append_log(job, "Planning…")
    plan = _chat(
        f"Write a short numbered plan only for this goal:\n{goal}",
        system="You are Atlas planner for JagX Bot. 3-6 short steps. No fluff.",
    ) or "1. Research\n2. Answer"
    _append_log(job, "Plan ready")

    # Step 2 — tools
    _append_log(job, "Running connectors / tools…")
    tool_text = _tools(goal)
    if tool_text:
        _append_log(job, "Tool data collected")
    else:
        _append_log(job, "No tool match (ok)")

    # Step 3 — final answer (Nimbus-style)
    _append_log(job, "Generating final answer…")
    prompt = (
        f"Goal:\n{goal}\n\nPlan:\n{plan}\n\n"
        f"{'Tool data:\n' + tool_text if tool_text else ''}\n\n"
        "Write a clear useful answer. If code is needed, put full files in markdown code fences."
    )
    answer = _chat(
        prompt,
        system=(
            "You are JagX Bot by JagX and JRILICENSE. "
            "Act like a capable coding agent. Be concrete. No ** bold stars."
        ),
    )
    if not answer and tool_text:
        answer = tool_text
    if not answer:
        job["status"] = "failed"
        job["result"] = "Bot could not generate an answer. Retry when the model is warm."
        _append_log(job, "Failed")
        job["updated_at"] = _now()
        return job

    # Step 4 — optional GitHub write (change your work)
    gh = _maybe_github_action(goal, plan, answer, job)
    if gh:
        _append_log(job, gh)
        answer = answer + "\n\n---\n" + gh

    # Step 5 — improve itself (append training + self-learn)
    _append_log(job, "Self-learn note saved")
    _self_learn_snippet(goal, answer)

    job["status"] = "done"
    job["result"] = answer[:15000]
    _append_log(job, "Done — safe to reopen app")
    job["updated_at"] = _now()
    return job


def _worker_loop() -> None:
    logger.info("JagX Bot job worker started")
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
            "github_write": bool(GITHUB_TOKEN.strip()),
            "default_repo": DEFAULT_REPO,
            "hint": "POST /jobs {goal,user_id}. Bot plans, uses tools, answers, optional GitHub, self-learns.",
        }

    start_worker()
    logger.info("JagX Bot job routes registered")
