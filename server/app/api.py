"""JagX v13 — FastAPI surface.

REST + SSE. The chat and council endpoints stream typed events that the UI
renders as the activity timeline.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from . import (
    agent, council, db, evals, goals, guard, llm, memory, sandbox, security,
    selfimprove, tools,
)
from .config import APP_NAME, VERSION, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("jagx.api")

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# auth dependency
# ---------------------------------------------------------------------------
def current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    payload = security.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    user = db.get_user(payload.get("sub", ""))
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return user


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class SignupBody(BaseModel):
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


class ChatBody(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    mode: str = "loop"  # loop | council
    model: str = ""


class ConversationPatch(BaseModel):
    title: str


class MemoryBody(BaseModel):
    content: str
    layer: str = "semantic"
    tags: str = ""


class GoalBody(BaseModel):
    title: str
    success_criterion: str = ""
    priority: int = 3
    budget: int = 20


class GoalPatch(BaseModel):
    title: Optional[str] = None
    success_criterion: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[int] = None
    budget: Optional[int] = None


class VariantBody(BaseModel):
    task_class: str
    name: str
    body: str


class EvalBody(BaseModel):
    name: str
    cases: list[dict] = Field(default_factory=list)


class ToolRunBody(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


class ApprovalBody(BaseModel):
    status: str  # approved | rejected


# ---------------------------------------------------------------------------
# health & meta
# ---------------------------------------------------------------------------
@router.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "app": APP_NAME,
        "version": VERSION,
        "llm": llm.health(),
        "sandbox": sandbox.status(),
        "embeddings": memory.stats("__probe__")["embedding_backend"],
        "jwt_secret_is_default": getattr(settings, "jwt_is_default", False),
    }


@router.get("/config")
def config() -> dict:
    return {
        "app": APP_NAME,
        "version": VERSION,
        "providers": settings.provider_status(),
        "local_first": True,
        "modes": ["loop", "council"],
        "tools": tools.describe(),
        "council": council.describe(),
        "guard": guard.describe(),
        "forge": selfimprove.describe(),
        "evals": evals.describe(),
        "goals": goals.describe(),
        "limits": {
            "max_steps": settings.max_steps,
            "max_tokens": settings.max_tokens,
            "critique_passes": settings.critique_passes,
        },
    }


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------
@router.post("/auth/signup")
def signup(body: SignupBody) -> dict:
    if not settings.allow_signup:
        raise HTTPException(status_code=403, detail="Signups are disabled.")
    email = (body.email or "").strip().lower()
    if "@" not in email or len(email) < 5:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="That email is already registered.")
    try:
        pw_hash = security.hash_password(body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    user = db.create_user(email, pw_hash)
    selfimprove.seed_variants(user["id"])
    evals.seed_suites(user["id"])
    return {"token": security.create_token(user["id"], user["email"]), "user": user}


@router.post("/auth/login")
def login(body: LoginBody) -> dict:
    user = db.get_user_by_email(body.email or "")
    if not user or not security.verify_password(body.password or "", user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return {
        "token": security.create_token(user["id"], user["email"]),
        "user": {"id": user["id"], "email": user["email"]},
    }


@router.get("/auth/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {"id": user["id"], "email": user["email"], "created_at": user["created_at"]}


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------
@router.get("/conversations")
def list_conversations(user: dict = Depends(current_user)) -> dict:
    return {"conversations": db.list_conversations(user["id"])}


@router.post("/conversations")
def create_conversation(user: dict = Depends(current_user)) -> dict:
    return db.create_conversation(user["id"])


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user: dict = Depends(current_user)) -> dict:
    conv = db.get_conversation(user["id"], conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation": conv, "messages": db.list_messages(conversation_id)}


@router.patch("/conversations/{conversation_id}")
def patch_conversation(conversation_id: str, body: ConversationPatch, user: dict = Depends(current_user)) -> dict:
    if not db.get_conversation(user["id"], conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db.rename_conversation(user["id"], conversation_id, body.title)
    return {"ok": True}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: dict = Depends(current_user)) -> dict:
    if not db.get_conversation(user["id"], conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db.delete_conversation(user["id"], conversation_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# chat (SSE)
# ---------------------------------------------------------------------------
def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _title_from(message: str) -> str:
    text = " ".join((message or "").split())
    return (text[:60] + "…") if len(text) > 60 else (text or "New task")


@router.post("/chat")
def chat(body: ChatBody, user: dict = Depends(current_user)) -> StreamingResponse:
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    user_settings = db.get_settings(user["id"])
    if body.model:
        user_settings["model"] = body.model

    conversation_id = body.conversation_id
    if conversation_id:
        if not db.get_conversation(user["id"], conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation_id = db.create_conversation(user["id"], _title_from(message))["id"]

    history = db.list_messages(conversation_id)
    db.add_message(conversation_id, "user", message)

    mode = body.mode if body.mode in ("loop", "council") else "loop"
    run_id = db.create_run(conversation_id, user["id"], mode=mode)

    # --- Forge loop 3: pick a prompt variant for this task class ---------
    task_class, fragment = selfimprove.variant_prompt(user["id"], message)
    variant = selfimprove.pick_variant(user["id"], task_class)
    variant_id = variant["id"] if variant else ""
    if fragment:
        base = user_settings.get("system_prompt") or ""
        user_settings["system_prompt"] = (base + "\n\nStyle guidance: " + fragment).strip()

    # --- Forge loop 2: experience replay --------------------------------
    try:
        exemplars = selfimprove.replay_examples(user["id"], message, limit=2)
        if exemplars:
            user_settings["system_prompt"] = (
                (user_settings.get("system_prompt") or "")
                + "\n\nApproaches that worked for similar tasks before:\n"
                + exemplars
            ).strip()
    except Exception:  # noqa: BLE001
        logger.exception("replay failed")

    def generate():
        yield _sse({"type": "conversation", "conversation_id": conversation_id, "title": _title_from(message)})
        yield _sse({"type": "meta", "run_id": run_id, "mode": mode, "task_class": task_class,
                    "variant": variant["name"] if variant else ""})

        answer = ""
        steps_used = 0
        tool_calls = 0
        tool_failures = 0
        critique: Optional[dict] = None
        provider = ""
        model_used = ""

        runner = council.run if mode == "council" else agent.run
        try:
            for event in runner(
                user_id=user["id"],
                conversation_id=conversation_id,
                user_message=message,
                history=history,
                user_settings=user_settings,
                run_id=run_id,
            ):
                if event.get("type") == "answer":
                    answer = event.get("text", "")
                elif event.get("type") == "tool_call":
                    tool_calls += 1
                elif event.get("type") == "tool_result":
                    if not event.get("ok"):
                        tool_failures += 1
                elif event.get("type") == "critique":
                    critique = event
                elif event.get("type") == "done":
                    steps_used = event.get("steps_used", 0)
                    provider = event.get("provider", "")
                    model_used = event.get("model", "")
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream failed")
            yield _sse({"type": "error", "message": f"Stream failed: {type(exc).__name__}"})

        # --- persist -----------------------------------------------------
        try:
            if answer:
                db.add_message(conversation_id, "assistant", answer)
                db.add_step(run_id, steps_used + 1, "answer", name="answer", output=answer)
            db.finish_run(
                run_id,
                status="done" if answer else "error",
                steps_used=steps_used,
                provider=provider,
                model=model_used,
            )
        except Exception:  # noqa: BLE001
            logger.exception("persist failed")

        # --- Forge loop 1 + 3 feedback -----------------------------------
        try:
            reflection = selfimprove.reflect(
                user["id"],
                run_id=run_id,
                question=message,
                answer=answer,
                steps_used=steps_used,
                tool_calls=tool_calls,
                tool_failures=tool_failures,
                critique=critique,
            )
            selfimprove.record_outcome(
                user["id"], task_class, variant_id, score=reflection["score"]
            )
            yield _sse({"type": "reflection", "score": reflection["score"], "parts": reflection["parts"]})
        except Exception:  # noqa: BLE001
            logger.exception("reflection failed")

        yield _sse({"type": "end"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/council")
def council_endpoint(body: ChatBody, user: dict = Depends(current_user)) -> StreamingResponse:
    body.mode = "council"
    return chat(body, user)


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
@router.get("/runs")
def list_runs(user: dict = Depends(current_user)) -> dict:
    return {"runs": db.list_runs(user["id"])}


@router.get("/runs/{run_id}")
def get_run(run_id: str, user: dict = Depends(current_user)) -> dict:
    run = db.get_run(user["id"], run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"run": run, "steps": db.list_steps(run_id)}


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
@router.get("/memories")
def list_memories(layer: str = "", user: dict = Depends(current_user)) -> dict:
    return {"memories": db.list_memories(user["id"], layer=layer), "stats": memory.stats(user["id"])}


@router.post("/memories")
def add_memory(body: MemoryBody, user: dict = Depends(current_user)) -> dict:
    if len((body.content or "").strip()) < 3:
        raise HTTPException(status_code=400, detail="Memory is too short.")
    return memory.remember(user["id"], body.content, layer=body.layer, tags=body.tags)


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, user: dict = Depends(current_user)) -> dict:
    db.delete_memory(user["id"], memory_id)
    return {"ok": True}


@router.post("/memories/consolidate")
def consolidate(user: dict = Depends(current_user)) -> dict:
    return memory.consolidate(user["id"])


# ---------------------------------------------------------------------------
# goals
# ---------------------------------------------------------------------------
@router.get("/goals")
def list_goals(status: str = "", user: dict = Depends(current_user)) -> dict:
    rows = db.list_goals(user["id"], status=status)
    for g in rows:
        g["checkpoints"] = db.list_checkpoints(g["id"])
    return {"goals": rows}


@router.post("/goals")
def create_goal(body: GoalBody, user: dict = Depends(current_user)) -> dict:
    if len((body.title or "").strip()) < 3:
        raise HTTPException(status_code=400, detail="Give the goal a title.")
    return goals.create(
        user["id"],
        body.title,
        success_criterion=body.success_criterion,
        priority=body.priority,
        budget=body.budget,
    )


@router.patch("/goals/{goal_id}")
def patch_goal(goal_id: str, body: GoalPatch, user: dict = Depends(current_user)) -> dict:
    if not db.get_goal(user["id"], goal_id):
        raise HTTPException(status_code=404, detail="Goal not found.")
    db.update_goal(user["id"], goal_id, **body.model_dump(exclude_none=True))
    return db.get_goal(user["id"], goal_id) or {}


@router.delete("/goals/{goal_id}")
def delete_goal(goal_id: str, user: dict = Depends(current_user)) -> dict:
    if not db.get_goal(user["id"], goal_id):
        raise HTTPException(status_code=404, detail="Goal not found.")
    db.delete_goal(user["id"], goal_id)
    return {"ok": True}


@router.post("/goals/{goal_id}/advance")
def advance_goal(goal_id: str, user: dict = Depends(current_user)) -> StreamingResponse:
    if not db.get_goal(user["id"], goal_id):
        raise HTTPException(status_code=404, detail="Goal not found.")

    def generate():
        for event in goals.advance(user["id"], goal_id):
            yield _sse(event)
        yield _sse({"type": "end"})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/goals/tick")
def tick_goals(user: dict = Depends(current_user)) -> dict:
    return {"advanced": goals.tick(user["id"])}


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------
@router.get("/approvals")
def list_approvals(status: str = "", user: dict = Depends(current_user)) -> dict:
    return {"approvals": db.list_approvals(user["id"], status=status)}


@router.post("/approvals/{approval_id}")
def decide_approval(approval_id: str, body: ApprovalBody, user: dict = Depends(current_user)) -> dict:
    if body.status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="status must be approved or rejected.")
    db.decide_approval(user["id"], approval_id, body.status)
    return {"ok": True}


# ---------------------------------------------------------------------------
# forge (self-improvement)
# ---------------------------------------------------------------------------
@router.get("/forge/stats")
def forge_stats(user: dict = Depends(current_user)) -> dict:
    return selfimprove.stats(user["id"])


@router.get("/forge/lessons")
def forge_lessons(user: dict = Depends(current_user)) -> dict:
    return {"lessons": db.list_lessons(user["id"])}


@router.get("/forge/variants")
def forge_variants(task_class: str = "", user: dict = Depends(current_user)) -> dict:
    selfimprove.seed_variants(user["id"])
    return {"variants": db.list_variants(user["id"], task_class=task_class)}


@router.post("/forge/variants")
def create_variant(body: VariantBody, user: dict = Depends(current_user)) -> dict:
    if not (body.name and body.body):
        raise HTTPException(status_code=400, detail="name and body are required.")
    return db.add_variant(user["id"], body.task_class or "general", body.name, body.body)


@router.post("/forge/optimize")
def forge_optimize(eval_id: str = "", user: dict = Depends(current_user)) -> dict:
    return selfimprove.optimize(user["id"], eval_id=eval_id)


@router.get("/forge/dataset")
def forge_dataset(fmt: str = "messages", limit: int = 200, user: dict = Depends(current_user)) -> dict:
    rows = selfimprove.export_dataset(user["id"], fmt=fmt, limit=limit)
    return {"format": fmt, "count": len(rows), "rows": rows}


@router.get("/forge/preferences")
def forge_preferences(limit: int = 200, user: dict = Depends(current_user)) -> dict:
    rows = selfimprove.export_preferences(user["id"], limit=limit)
    return {"count": len(rows), "rows": rows}


# ---------------------------------------------------------------------------
# evals
# ---------------------------------------------------------------------------
@router.get("/evals")
def list_evals(user: dict = Depends(current_user)) -> dict:
    evals.seed_suites(user["id"])
    return {"evals": db.list_evals(user["id"])}


@router.post("/evals")
def create_eval(body: EvalBody, user: dict = Depends(current_user)) -> dict:
    if not body.name:
        raise HTTPException(status_code=400, detail="name is required.")
    return db.create_eval(user["id"], body.name, body.cases)


@router.post("/evals/{eval_id}/run")
def run_eval(eval_id: str, variant_id: str = "", user: dict = Depends(current_user)) -> dict:
    return evals.run_eval(user["id"], eval_id, variant_id=variant_id)


@router.post("/evals/{eval_id}/compare")
def compare_eval(eval_id: str, user: dict = Depends(current_user)) -> dict:
    return evals.compare_variants(user["id"], eval_id)


@router.get("/evals/{eval_id}/results")
def eval_results(eval_id: str, user: dict = Depends(current_user)) -> dict:
    return {"results": db.list_eval_results(eval_id)}


# ---------------------------------------------------------------------------
# tools & settings
# ---------------------------------------------------------------------------
@router.get("/tools")
def list_tools(user: dict = Depends(current_user)) -> dict:
    return {"tools": tools.describe(), "sandbox": sandbox.status()}


@router.post("/tools/run")
def run_tool(body: ToolRunBody, user: dict = Depends(current_user)) -> dict:
    verdict = guard.check(body.name, json.dumps(body.arguments)[:800], tool=body.name)
    if verdict.verdict != guard.ALLOW:
        return {"ok": False, "guard": verdict.to_dict(), "output": f"Held: {verdict.reason}"}
    output, ok = tools.execute(body.name, body.arguments, user_id=user["id"])
    return {"ok": ok, "output": output, "guard": verdict.to_dict()}


@router.get("/settings")
def get_settings(user: dict = Depends(current_user)) -> dict:
    return db.get_settings(user["id"])


@router.put("/settings")
def put_settings(values: dict, user: dict = Depends(current_user)) -> dict:
    return db.put_settings(user["id"], values)


# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    db.init_db()
    app = FastAPI(title=APP_NAME, version=VERSION, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=r"https?://.*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # The multimodal surface lives in its own router (media_api.py) to keep this
    # module readable; it is mounted here so create_app() always returns the
    # complete API.
    from .media_api import router as media_router

    app.include_router(media_router)

    @app.get("/")
    def root() -> dict:
        return {"app": APP_NAME, "version": VERSION, "docs": "/api/docs", "health": "/api/health"}

    return app
