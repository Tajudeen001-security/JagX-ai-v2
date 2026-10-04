"""
JagX sandbox workspace — run code, files, network fetch, GitHub import/export.
Created by JagX & JRILICENSE.

Safety limits (shared free server):
- Python only, 8s timeout, 64KB stdout
- Workspace under /tmp/jagx_ws/<user_id>
- Network: GET only, 1MB max, no private IP targets
- GitHub uses per-user vault token (never returned)
"""
from __future__ import annotations

import ast
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger("jagx-ai")
HTTP = requests.Session()

PROJ = os.environ.get("JAGX_PROJECTS_FILE", "jagx_projects.json")
WS_ROOT = Path(os.environ.get("JAGX_WS_ROOT", "/tmp/jagx_ws"))
MAX_CODE = 20000
MAX_STDOUT = 65536
RUN_TIMEOUT = 8
MAX_FETCH = 1_000_000
_lock = threading.Lock()

SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "len": len,
    "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(user_id: str) -> str:
    u = re.sub(r"[^a-zA-Z0-9_-]", "", (user_id or "guest"))[:40] or "guest"
    return u


def _ws(user_id: str) -> Path:
    p = WS_ROOT / _uid(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_path(user_id: str, rel: str) -> Optional[Path]:
    rel = (rel or "").replace("..", "").lstrip("/")
    if not rel or rel.startswith("."):
        return None
    base = _ws(user_id).resolve()
    target = (base / rel).resolve()
    if not str(target).startswith(str(base)):
        return None
    return target


def safe_calc(expr: str) -> dict:
    expr = (expr or "").strip()
    if not expr or len(expr) > 300:
        return {"ok": False, "error": "expression required (max 300 chars)"}
    if any(bad in expr for bad in ("import", "__", "open", "exec", "eval", "os", "sys")):
        return {"ok": False, "error": "not allowed"}
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return {"ok": False, "error": f"syntax: {e}"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp,
                             ast.Expression, ast.Load, ast.Add, ast.Sub, ast.Mult,
                             ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub,
                             ast.UAdd, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
                             ast.Gt, ast.LtE, ast.GtE)):
            continue
        if isinstance(node, ast.Attribute):
            return {"ok": False, "error": "attributes not allowed"}
        return {"ok": False, "error": "only math expressions"}
    try:
        val = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, SAFE_NAMES)
        return {"ok": True, "result": val}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def run_python(user_id: str, code: str, filename: str = "main.py") -> dict:
    """Write code into workspace and run with timeout to confirm it works."""
    code = code or ""
    if not code.strip():
        return {"ok": False, "error": "code required"}
    if len(code) > MAX_CODE:
        return {"ok": False, "error": f"code max {MAX_CODE} chars"}
    path = _safe_path(user_id, filename or "main.py")
    if path is None:
        return {"ok": False, "error": "bad filename"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            ["python", str(path)],
            cwd=str(_ws(user_id)),
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(_ws(user_id)),
                "PYTHONPATH": str(_ws(user_id)),
            },
        )
        out = (proc.stdout or "")[:MAX_STDOUT]
        err = (proc.stderr or "")[:MAX_STDOUT]
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": out,
            "stderr": err,
            "file": str(path.relative_to(_ws(user_id))),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {RUN_TIMEOUT}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def list_files(user_id: str) -> dict:
    base = _ws(user_id)
    files = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(base))
            files.append({"path": rel, "size": p.stat().st_size})
    return {"ok": True, "files": files[:200]}


def write_file(user_id: str, path: str, content: str) -> dict:
    target = _safe_path(user_id, path)
    if target is None:
        return {"ok": False, "error": "bad path"}
    content = content or ""
    if len(content) > 200_000:
        return {"ok": False, "error": "file too large"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}


def read_file(user_id: str, path: str) -> dict:
    target = _safe_path(user_id, path)
    if target is None or not target.exists() or not target.is_file():
        return {"ok": False, "error": "not found"}
    if target.stat().st_size > 200_000:
        return {"ok": False, "error": "file too large to return"}
    return {"ok": True, "path": path, "content": target.read_text(encoding="utf-8", errors="replace")}


def delete_file(user_id: str, path: str) -> dict:
    target = _safe_path(user_id, path)
    if target is None or not target.exists():
        return {"ok": False, "error": "not found"}
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"ok": True, "deleted": path}


def clear_workspace(user_id: str) -> dict:
    base = _ws(user_id)
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "cleared": True}


def _is_public_http(url: str) -> bool:
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            return False
        host = (u.hostname or "").lower()
        if not host or host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            return False
        if host.startswith("10.") or host.startswith("192.168.") or host.startswith("169.254."):
            return False
        if re.match(r"^172\.(1[6-9]|2\d|3[0-1])\.", host):
            return False
        return True
    except Exception:
        return False


def fetch_url(url: str) -> dict:
    url = (url or "").strip()
    if not _is_public_http(url):
        return {"ok": False, "error": "only public http(s) URLs allowed"}
    try:
        r = HTTP.get(url, timeout=15, stream=True, headers={"User-Agent": "JagX-Sandbox/1.0"})
        data = b""
        for chunk in r.iter_content(8192):
            data += chunk
            if len(data) > MAX_FETCH:
                return {"ok": False, "error": "response too large"}
        text = data.decode("utf-8", errors="replace")
        return {
            "ok": True,
            "status": r.status_code,
            "content_type": r.headers.get("content-type", ""),
            "text": text[:50000],
            "bytes": len(data),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def _token_for(user_id: str, explicit: Optional[str] = None) -> Optional[str]:
    if explicit and str(explicit).strip() and str(explicit).strip() != "***":
        return str(explicit).strip()
    try:
        import jagx_vault as v
        return v.get_secret(user_id, "github")
    except Exception:
        pass
    env = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    return env.strip() or None


def github_import(user_id: str, repo: str, path: str, token: Optional[str] = None, ref: str = "main") -> dict:
    """Pull a file from GitHub into the workspace before working on it."""
    tok = _token_for(user_id, token)
    repo = (repo or "").strip()
    path = (path or "").lstrip("/")
    if repo.count("/") != 1 or not path:
        return {"ok": False, "error": "repo must be owner/name and path required"}
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "JagX-Sandbox"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        r = HTTP.get(api, headers=headers, params={"ref": ref or "main"}, timeout=25)
        if r.status_code != 200:
            return {"ok": False, "error": f"GitHub {r.status_code}: {r.text[:200]}"}
        data = r.json()
        if isinstance(data, list):
            names = [x.get("path") for x in data if isinstance(x, dict)]
            return {"ok": True, "type": "dir", "entries": names[:100]}
        content_b64 = data.get("content") or ""
        raw = base64.b64decode(content_b64.replace("\n", "")).decode("utf-8", errors="replace")
        local_name = path.split("/")[-1]
        write_file(user_id, local_name, raw)
        # also keep full relative path if nested
        if "/" in path:
            write_file(user_id, path, raw)
        return {
            "ok": True,
            "type": "file",
            "repo": repo,
            "path": path,
            "local": local_name,
            "bytes": len(raw),
            "preview": raw[:2000],
            "sha": data.get("sha"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def github_export(user_id: str, repo: str, path: str, local_path: str = "", message: str = "", token: Optional[str] = None) -> dict:
    """Push a workspace file back to the user's GitHub."""
    tok = _token_for(user_id, token)
    if not tok:
        return {"ok": False, "error": "No GitHub token. Save one via POST /vault first."}
    repo = (repo or "").strip()
    path = (path or "").lstrip("/")
    local_path = (local_path or path).lstrip("/")
    if repo.count("/") != 1 or not path:
        return {"ok": False, "error": "repo owner/name and path required"}
    target = _safe_path(user_id, local_path)
    if target is None or not target.exists():
        return {"ok": False, "error": f"local file missing: {local_path}"}
    content = target.read_text(encoding="utf-8", errors="replace")
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "JagX-Sandbox",
    }
    sha = None
    try:
        gr = HTTP.get(api, headers=headers, timeout=20)
        if gr.status_code == 200:
            sha = gr.json().get("sha")
    except Exception:
        pass
    body = {
        "message": (message or f"JagX sandbox export {path}")[:200],
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": "main",
    }
    if sha:
        body["sha"] = sha
    try:
        r = HTTP.put(api, headers=headers, json=body, timeout=40)
        if r.status_code in (200, 201):
            return {"ok": True, "note": f"Exported to {repo}/{path}"}
        return {"ok": False, "error": f"GitHub {r.status_code}: {r.text[:240]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def _load_proj() -> dict:
    if not os.path.exists(PROJ):
        return {}
    try:
        with open(PROJ, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_proj(data: dict) -> None:
    with open(PROJ, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def save_project(user_id: str, name: str, notes: str) -> dict:
    user_id = _uid(user_id)
    item = {
        "id": str(uuid.uuid4()),
        "name": (name or "untitled")[:80],
        "notes": (notes or "")[:8000],
        "ts": _now(),
    }
    with _lock:
        data = _load_proj()
        rows = data.get(user_id) or []
        rows.insert(0, item)
        data[user_id] = rows[:50]
        _save_proj(data)
    return {"ok": True, "project": item}


def list_projects(user_id: str) -> dict:
    with _lock:
        rows = list(_load_proj().get(_uid(user_id)) or [])
    return {"projects": rows}


def register_sandbox_routes(app=None) -> None:
    app = app or globals().get("app")
    if app is None:
        return
    from fastapi import Body

    @app.post("/calc")
    def calc_ep(body: dict = Body(...)):
        return safe_calc(str(body.get("expr") or body.get("expression") or ""))

    @app.post("/sandbox/run")
    def run_ep(body: dict = Body(...)):
        return run_python(
            str(body.get("user_id") or "guest"),
            str(body.get("code") or ""),
            str(body.get("filename") or "main.py"),
        )

    @app.get("/sandbox/files")
    def files_ep(user_id: str = "guest"):
        return list_files(user_id)

    @app.post("/sandbox/write")
    def write_ep(body: dict = Body(...)):
        return write_file(
            str(body.get("user_id") or "guest"),
            str(body.get("path") or ""),
            str(body.get("content") or ""),
        )

    @app.post("/sandbox/read")
    def read_ep(body: dict = Body(...)):
        return read_file(str(body.get("user_id") or "guest"), str(body.get("path") or ""))

    @app.post("/sandbox/delete")
    def del_ep(body: dict = Body(...)):
        return delete_file(str(body.get("user_id") or "guest"), str(body.get("path") or ""))

    @app.post("/sandbox/clear")
    def clear_ep(body: dict = Body(...)):
        return clear_workspace(str(body.get("user_id") or "guest"))

    @app.post("/sandbox/fetch")
    def fetch_ep(body: dict = Body(...)):
        return fetch_url(str(body.get("url") or ""))

    @app.post("/sandbox/github_import")
    def gh_in(body: dict = Body(...)):
        # optional token accepted once; prefer vault
        user_id = str(body.get("user_id") or "guest")
        token = body.get("token")
        if token:
            try:
                import jagx_vault as v
                v.put_secret(user_id, "github", str(token))
            except Exception:
                pass
        return github_import(
            user_id,
            str(body.get("repo") or ""),
            str(body.get("path") or ""),
            token=None,
            ref=str(body.get("ref") or "main"),
        )

    @app.post("/sandbox/github_export")
    def gh_out(body: dict = Body(...)):
        user_id = str(body.get("user_id") or "guest")
        token = body.get("token")
        if token:
            try:
                import jagx_vault as v
                v.put_secret(user_id, "github", str(token))
            except Exception:
                pass
        return github_export(
            user_id,
            str(body.get("repo") or ""),
            str(body.get("path") or ""),
            local_path=str(body.get("local_path") or body.get("path") or ""),
            message=str(body.get("message") or ""),
            token=None,
        )

    @app.post("/projects")
    def proj_put(body: dict = Body(...)):
        return save_project(str(body.get("user_id") or "guest"), str(body.get("name") or ""), str(body.get("notes") or ""))

    @app.get("/projects")
    def proj_list(user_id: str = "guest"):
        return list_projects(user_id)

    @app.get("/sandbox/info")
    def sb_info():
        return {
            "run": "POST /sandbox/run {user_id, code, filename?}",
            "files": "GET /sandbox/files?user_id=",
            "write": "POST /sandbox/write {user_id, path, content}",
            "read": "POST /sandbox/read {user_id, path}",
            "fetch": "POST /sandbox/fetch {url}",
            "github_import": "POST /sandbox/github_import {user_id, repo, path}",
            "github_export": "POST /sandbox/github_export {user_id, repo, path, local_path?}",
            "limits": {"timeout_s": RUN_TIMEOUT, "code_chars": MAX_CODE, "fetch_bytes": MAX_FETCH},
        }

    @app.get("/ready")
    def ready():
        return {
            "ok": True,
            "features": [
                "chat", "news", "geo", "weather", "files",
                "mcp", "jobs", "vault", "code", "memory", "calc", "projects",
                "sandbox_run", "sandbox_files", "sandbox_fetch",
                "github_import", "github_export",
            ],
        }

    logger.info("JagX sandbox expanded: run/files/fetch/github_import/export")
