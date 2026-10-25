"""FastAPI application factory — JagX AI v9.8 (multi-agent default chat)."""

from __future__ import annotations

import logging
import time
import hashlib
import json
import os
import threading
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from jagx_api import __version__, auth, knowledge, memory_store, conversation_store, engineering, video_generation
from jagx_api.chat import generate_response, stream_response
from jagx_api.jobs_api import router as jobs_router
from jagx_api.autonomy import router as autonomy_router
from jagx_api.agents_api import router as agents_router
from jagx_api.preferences_api import router as preferences_router
from jagx_api.training_api import router as training_router
from jagx_api import preferences as prefs_mod
from jagx_api import job_worker
from jagx_api.config import settings
from jagx_api.providers import provider_status, provider_health
from jagx_api.tools import safe_calc, run_tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jagx-ai")
APP_START = time.time()
_public_chat_store: dict = {}
_public_lock = threading.Lock()


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > settings.max_body_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        return await call_next(request)


class GlobalIPRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        if not auth.check_ip_rate(ip):
            return JSONResponse(status_code=429, content={"detail": "Too many requests from this IP"})
        return await call_next(request)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    history: Optional[List[dict]] = None
    stream: bool = False
    stream_format: str = "sse"
    conversation_id: Optional[str] = None
    training_consent: bool = False
    mode: str = Field(default="agents", pattern="^(fast|agents|reason)$")
    reflect: bool = False


class CreateKeyRequest(BaseModel):
    owner_label: str = "anonymous"
    admin_secret: str
    tier: str = "free"


class MemoryRequest(BaseModel):
    text: str = ""
    tags: str = ""
    query: str = ""
    memory_id: str = ""
    user_id: str = "guest"
    action: str = "remember"
    consent: bool = False


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New conversation", max_length=160)


class CalcRequest(BaseModel):
    expression: str


class ToolRequest(BaseModel):
    tool: str
    input: Optional[dict] = None


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    duration: int = Field(default=4, ge=2, le=8)
    model: str = "google/veo-3.1-fast"
    aspect_ratio: str = "16:9"


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1200)
    width: int = Field(default=768, ge=256, le=1024)
    height: int = Field(default=768, ge=256, le=1024)


def create_app() -> FastAPI:
    app = FastAPI(
        title="JagX AI",
        version=__version__,
        description="Text + coding AI API with multi-agent core. Created by JagX & JRILICENSE.",
    )
    app.add_middleware(GlobalIPRateLimitMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)

    _default_origins = [
        "https://jagxai.name.ng", "https://www.jagxai.name.ng", "https://jagx-ai-app.vercel.app",
        "http://localhost:3000", "http://localhost:5173", "http://localhost:8080",
    ]
    cors_env = os.environ.get("JAGX_CORS_ORIGINS", "")
    origins = [o.strip().rstrip("/") for o in cors_env.split(",") if o.strip()] if cors_env.strip() else _default_origins
    if os.environ.get("JAGX_CORS_ALLOW_ALL", "").lower() in ("1", "true", "yes"):
        origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https://jagx-ai-[a-z0-9-]+-jagxs-projects\.vercel\.app",
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "x-api-key", "x-admin-secret"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.include_router(jobs_router)
    app.include_router(autonomy_router)
    app.include_router(agents_router)
    app.include_router(preferences_router)
    app.include_router(training_router)

    @app.on_event("startup")
    def _startup() -> None:
        n = knowledge.load_brain(".")
        logger.info("JagX AI %s started — brain=%s providers=%s", __version__, n, provider_status())
        job_worker.start()

    def _require_key(x_api_key: Optional[str]) -> str:
        key = (x_api_key or "").strip()
        if not auth.is_valid_key(key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        ok, info = auth.check_rate_limit(key)
        if not ok:
            raise HTTPException(status_code=429, detail=info)
        return key

    @app.on_event("shutdown")
    def _shutdown() -> None:
        job_worker.stop()

    @app.get("/")
    def root():
        return {
            "status": f"JagX AI {__version__} is running",
            "version": __version__,
            "created_by": "JagX & JRILICENSE",
            "features": [
                "multi_agent_default_chat", "permanent_api_keys", "public_chat", "multi_provider",
                "streaming", "memory", "style_preferences", "calc", "tools", "fetch_url",
                "rate_limits", "persistent_conversations", "durable_background_jobs",
                "image_generation_api", "agents_api", "multi_agent_streaming", "lora_flywheel",
            ],
            "docs": "/docs",
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok", "service": "jagx-ai-v2", "version": __version__,
            "uptime_seconds": int(time.time() - APP_START),
            "local_brain": knowledge.size(), "providers": provider_status(),
        }

    @app.post("/public-chat")
    def public_chat(req: ChatRequest, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        rpm = settings.public_chat_rpm
        with _public_lock:
            recent = [t for t in _public_chat_store.get(client_ip, []) if now - t < 60]
            if len(recent) >= rpm:
                raise HTTPException(status_code=429, detail=f"Guest limit reached ({rpm}/min)")
            recent.append(now)
            _public_chat_store[client_ip] = recent
        msg = req.message.strip()
        if not msg:
            raise HTTPException(status_code=400, detail="Empty message")
        history = []
        for item in (req.history or [])[-8:]:
            if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
                history.append({"role": item["role"], "content": str(item.get("content") or "")[:3000]})
        if req.stream:
            def sse_gen():
                try:
                    for piece in stream_response(msg, history):
                        yield f"data: {json.dumps({'token': piece}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception:
                    logger.exception("public stream failed")
                    yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
            return StreamingResponse(sse_gen(), media_type="text/event-stream; charset=utf-8",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        result = generate_response(msg, history, use_tools=True, training_consent=False)
        out = {"response": result["response"], "provider": result.get("provider"),
               "local_hit": result.get("local_hit", False),
               "rate_limit": f"{max(0, rpm - len(recent))} guest requests remaining this minute",
               "version": __version__}
        if result.get("attachment"):
            out["attachment"] = result["attachment"]
        return out

    @app.post("/chat")
    def chat(req: ChatRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = _require_key(x_api_key)
        msg = req.message.strip()
        if not msg:
            raise HTTPException(status_code=400, detail="Empty message")
        history = req.history
        if req.conversation_id:
            saved = conversation_store.get(key, req.conversation_id, limit=settings.max_history)
            if saved is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            history = [{"role": m["role"], "content": m["content"]} for m in saved["messages"]]
            if not conversation_store.append(key, req.conversation_id, "user", msg):
                raise HTTPException(status_code=404, detail="Conversation not found")
        memory_hits = memory_store.recall(hashlib.sha256(("jagx-memory-v1:" + key).encode()).hexdigest(), msg)
        memory_context = "\n".join(str(hit.get("text", "")) for hit in memory_hits) if memory_hits else None
        pref_owner = hashlib.sha256(("jagx-prefs-v1:" + key).encode()).hexdigest()
        pref_ctx = prefs_mod.format_preference_context(pref_owner)
        if pref_ctx:
            memory_context = (memory_context + "\n" + pref_ctx) if memory_context else pref_ctx
        if req.stream:
            mode = (req.mode or "agents").lower()
            if mode in ("agents", "reason"):
                def ma_sse_gen():
                    chunks = []
                    try:
                        from jagx_api.agent_stream import stream_multi_agent_sse
                        for event in stream_multi_agent_sse(
                            msg,
                            history=history,
                            memory_context=memory_context,
                            reason=(mode == "reason"),
                            reflect=bool(req.reflect),
                        ):
                            yield event
                            if event.startswith("data: "):
                                try:
                                    payload = json.loads(event[6:].strip())
                                    if payload.get("event") == "token" and payload.get("token"):
                                        chunks.append(payload["token"])
                                except Exception:
                                    pass
                        if req.conversation_id and chunks:
                            conversation_store.append(key, req.conversation_id, "assistant", "".join(chunks))
                    except Exception:
                        logger.exception("multi-agent stream failed; falling back")
                        try:
                            for piece in stream_response(msg, history, memory_context):
                                chunks.append(piece)
                                yield f"data: {json.dumps({'event': 'token', 'token': piece}, ensure_ascii=False)}\n\n"
                            yield f"data: {json.dumps({'event': 'done'})}\n\n"
                            if req.conversation_id and chunks:
                                conversation_store.append(key, req.conversation_id, "assistant", "".join(chunks))
                        except Exception:
                            yield f"data: {json.dumps({'event': 'error', 'detail': 'stream failed'})}\n\n"
                            yield f"data: {json.dumps({'event': 'done'})}\n\n"
                return StreamingResponse(ma_sse_gen(), media_type="text/event-stream; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
            def sse_gen():
                chunks = []
                try:
                    for piece in stream_response(msg, history, memory_context):
                        chunks.append(piece)
                        yield f"data: {json.dumps({'token': piece}, ensure_ascii=False)}\n\n"
                    if req.conversation_id and chunks:
                        conversation_store.append(key, req.conversation_id, "assistant", "".join(chunks))
                    yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception:
                    logger.exception("stream failed")
                    yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
            return StreamingResponse(sse_gen(), media_type="text/event-stream; charset=utf-8",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        mode = (req.mode or "agents").lower()
        if mode in ("agents", "reason"):
            try:
                from jagx_api.agents import run_multi_agent
                ma = run_multi_agent(
                    msg, history=history, memory_context=memory_context,
                    reason=(mode == "reason"), reflect=bool(req.reflect),
                )
                result = {
                    "response": ma.get("response"),
                    "provider": ma.get("provider", "multi-agent"),
                    "local_hit": False,
                    "attachment": ma.get("attachment"),
                    "agents_used": ma.get("agents_used"),
                }
            except Exception:
                logger.exception("multi-agent chat failed; falling back to fast path")
                result = generate_response(msg, history, memory_context=memory_context, training_consent=req.training_consent)
        else:
            result = generate_response(msg, history, memory_context=memory_context, training_consent=req.training_consent)
        if req.conversation_id:
            conversation_store.append(key, req.conversation_id, "assistant", result.get("response", ""), result.get("provider"))
        info = auth.key_info(key)
        out = {"response": result["response"], "provider": result.get("provider"),
               "local_hit": result.get("local_hit", False),
               "rate_limit": f"tier={info.get('tier')}" if info else "ok", "version": __version__}
        if result.get("attachment"):
            out["attachment"] = result["attachment"]
        if result.get("agents_used"):
            out["agents_used"] = result["agents_used"]
        return out

    @app.post("/create-key")
    def create_key(req: CreateKeyRequest):
        result = auth.create_key(req.owner_label, req.tier, req.admin_secret)
        if not result.get("ok"):
            raise HTTPException(status_code=403, detail=result.get("error", "forbidden"))
        return result

    @app.get("/keys/me")
    def keys_me(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = (x_api_key or "").strip()
        info = auth.key_info(key)
        if not info:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return info

    @app.post("/memory")
    def memory(req: MemoryRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = _require_key(x_api_key)
        owner = hashlib.sha256(("jagx-memory-v1:" + key).encode()).hexdigest()
        action = (req.action or "remember").lower()
        if action == "remember":
            if not req.consent:
                raise HTTPException(status_code=400, detail="consent=true is required to store memory")
            mid = memory_store.remember(owner, req.text, req.tags)
            return {"ok": True, "memory_id": mid}
        if action == "recall":
            hits = memory_store.recall(owner, req.query or req.text)
            return {"ok": True, "memories": hits}
        if action == "forget":
            ok = memory_store.forget(owner, req.memory_id)
            return {"ok": ok}
        raise HTTPException(status_code=400, detail="action must be remember|recall|forget")

    @app.post("/calc")
    def calc(req: CalcRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return safe_calc(req.expression)

    @app.post("/conversations")
    def create_conversation(req: ConversationCreateRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = _require_key(x_api_key)
        return {"ok": True, "conversation": conversation_store.create(key, req.title)}

    @app.get("/conversations")
    def list_conversations(limit: int = 50, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = _require_key(x_api_key)
        return {"ok": True, "conversations": conversation_store.list_for(key, limit)}

    @app.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, limit: int = 100, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = _require_key(x_api_key)
        item = conversation_store.get(key, conversation_id, limit)
        if item is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"ok": True, "conversation": item}

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = _require_key(x_api_key)
        if not conversation_store.delete(key, conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"ok": True, "deleted": True}

    @app.post("/images/generate")
    def generate_image(req: ImageGenerationRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        from jagx_api.image_generation import generate
        try:
            result = generate(req.prompt, req.width, req.height)
            return {"ok": True, **result}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/video/generate")
    def generate_video(req: VideoGenerationRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        try:
            return video_generation.generate(req.prompt, req.duration, req.model, req.aspect_ratio)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.warning("video generation failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Video generation failed or is unavailable.")

    @app.post("/tool")
    def tool_endpoint(req: ToolRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        text, attachment = run_tool(req.tool, req.input or {})
        out = {"ok": True, "result": text}
        if attachment:
            out["attachment"] = attachment
        return out

    @app.get("/providers/health")
    def providers_health(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return {"providers": provider_health(), "note": "Reachability does not validate credentials, quota, or model access."}

    @app.get("/engineering/capabilities")
    def engineering_capabilities(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return engineering.capability_map()

    return app


app = create_app()
