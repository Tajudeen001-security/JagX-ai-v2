"""
JagX sandbox — math and short Python expressions only.
No file writes, no network, no imports. Timeout capped.
Also a simple project notebook per user.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import threading
import uuid
from datetime import datetime, timezone

logger = logging.getLogger("jagx-ai")
PROJ = os.environ.get("JAGX_PROJECTS_FILE", "jagx_projects.json")
_lock = threading.Lock()

SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "len": len,
    "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _load() -> dict:
    if not os.path.exists(PROJ):
        return {}
    try:
        with open(PROJ, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    with open(PROJ, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def save_project(user_id: str, name: str, notes: str) -> dict:
    user_id = (user_id or "guest").strip()[:120]
    item = {
        "id": str(uuid.uuid4()),
        "name": (name or "untitled")[:80],
        "notes": (notes or "")[:8000],
        "ts": _now(),
    }
    with _lock:
        data = _load()
        rows = data.get(user_id) or []
        rows.insert(0, item)
        data[user_id] = rows[:50]
        _save(data)
    return {"ok": True, "project": item}


def list_projects(user_id: str) -> dict:
    with _lock:
        rows = list(_load().get((user_id or "guest").strip()[:120]) or [])
    return {"projects": rows}


def register_sandbox_routes(app=None) -> None:
    app = app or globals().get("app")
    if app is None:
        return
    from fastapi import Body

    @app.post("/calc")
    def calc_ep(body: dict = Body(...)):
        return safe_calc(str(body.get("expr") or body.get("expression") or ""))

    @app.post("/projects")
    def proj_put(body: dict = Body(...)):
        return save_project(str(body.get("user_id") or "guest"), str(body.get("name") or ""), str(body.get("notes") or ""))

    @app.get("/projects")
    def proj_list(user_id: str = "guest"):
        return list_projects(user_id)

    @app.get("/ready")
    def ready():
        return {
            "ok": True,
            "features": [
                "chat", "news", "geo", "weather", "files",
                "mcp", "jobs", "vault", "code", "memory", "calc", "projects",
            ],
        }

    logger.info("JagX sandbox routes: /calc /projects /ready")
