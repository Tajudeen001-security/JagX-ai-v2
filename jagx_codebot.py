"""
JagX code bot — write real files to ANY user's GitHub.
Token comes from that user's vault (never logged, never returned).
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import threading
from typing import Optional

import requests

logger = logging.getLogger("jagx-ai")
HTTP = requests.Session()


def _token_for(user_id: str, explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit.strip() and explicit.strip() != "***":
        return explicit.strip()
    try:
        import jagx_vault as v
        return v.get_secret(user_id, "github")
    except Exception:
        return None


def github_put(user_id: str, repo: str, path: str, content: str, message: str, token: Optional[str] = None) -> str:
    tok = _token_for(user_id, token)
    if not tok:
        return "No GitHub token for this user. Save one in Vault first (Connectors → GitHub)."
    repo = (repo or "").strip()
    if repo.count("/") != 1:
        return "repo must look like owner/name"
    path = path.lstrip("/")
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {tok}",
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
        "message": (message or "JagX Bot update")[:200],
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": "main",
    }
    if sha:
        body["sha"] = sha
    r = HTTP.put(api, headers=headers, json=body, timeout=40)
    if r.status_code in (200, 201):
        return f"Wrote {repo}/{path}"
    return f"GitHub {r.status_code}: {r.text[:240]}"


def _chat(prompt: str) -> str:
    try:
        port = os.environ.get("PORT", "10000")
        r = HTTP.post(
            f"http://127.0.0.1:{port}/chat",
            json={"message": prompt},
            timeout=120,
        )
        if r.status_code == 200:
            return str(r.json().get("response") or "")
    except Exception as e:
        logger.info("codebot chat: %s", e)
    return ""


def build_and_push(user_id: str, goal: str, repo: str, token: Optional[str] = None) -> dict:
    """Ask the model for files, then commit them to the user's repo."""
    prompt = (
        "You are JagX Bot. Build what the user asked.\n"
        "Reply ONLY with file blocks in this exact format (one or more):\n"
        "FILE: path/to/file.ext\n"
        "```\n"
        "file contents\n"
        "```\n\n"
        f"Goal:\n{goal}\n"
        "Prefer a small working app (HTML or Python) if they did not name a stack."
    )
    raw = _chat(prompt)
    if not raw:
        return {"ok": False, "error": "model returned nothing"}
    blocks = re.findall(r"FILE:\s*([^\n]+)\n```(?:\w+)?\n([\s\S]*?)```", raw)
    if not blocks:
        # fallback: one markdown note so the work is not lost
        blocks = [("BOT_BUILD.md", f"# Build\n\nGoal: {goal}\n\n{raw}")]
    notes = []
    for path, content in blocks[:8]:
        path = path.strip().replace("..", "")
        if not path or path.startswith("/"):
            continue
        notes.append(github_put(user_id, repo, path, content, f"JagX Bot: {goal[:70]}", token))
    return {"ok": True, "files": notes, "preview": raw[:1500]}


def register_code_routes(app=None) -> None:
    app = app or globals().get("app")
    if app is None:
        return
    from fastapi import Body

    @app.post("/code/push")
    def code_push(body: dict = Body(...)):
        # token in body is accepted once then should be vaulted; never echo it
        user_id = str(body.get("user_id") or "guest")
        token = body.get("token")
        if token:
            try:
                import jagx_vault as v
                v.put_secret(user_id, "github", str(token))
            except Exception:
                pass
        note = github_put(
            user_id,
            str(body.get("repo") or ""),
            str(body.get("path") or ""),
            str(body.get("content") or ""),
            str(body.get("message") or "JagX Bot"),
        )
        return {"ok": note.startswith("Wrote"), "note": note}

    @app.post("/code/build")
    def code_build(body: dict = Body(...)):
        user_id = str(body.get("user_id") or "guest")
        goal = str(body.get("goal") or "").strip()
        repo = str(body.get("repo") or "").strip()
        token = body.get("token")
        if token:
            try:
                import jagx_vault as v
                v.put_secret(user_id, "github", str(token))
            except Exception:
                pass
        if not goal or not repo:
            return {"ok": False, "error": "goal and repo (owner/name) required"}

        def _run():
            try:
                build_and_push(user_id, goal, repo)
            except Exception as e:
                logger.warning("code build: %s", e)

        threading.Thread(target=_run, name="jagx-codebot", daemon=True).start()
        return {
            "ok": True,
            "status": "building",
            "note": "Writing files to your GitHub in the background. Token is not stored in the reply.",
        }

    logger.info("JagX code bot routes: POST /code/push /code/build")
