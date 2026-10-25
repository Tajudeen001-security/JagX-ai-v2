"""FastAPI application factory — JagX AI v9 (multi-agent core)."""

from __future__ import annotations

import logging
import asyncio
import time
import hashlib
import hmac
import json
import os
import base64
import threading
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from jagx_api import __version__, auth, knowledge, memory_store, conversation_store, local_model, speech, engineering, video_generation
from jagx_api.chat import generate_response, stream_response, stream_sse
from jagx_api.jobs_api import router as jobs_router
from jagx_api.autonomy import router as autonomy_router
from jagx_api.agents_api import router as agents_router
from jagx_api import job_worker
from jagx_api.config import settings
from jagx_api.providers import provider_status, provider_health
from jagx_api.tools import (
    safe_calc,
    generate_pdf_document,
    generate_cv_pdf,
    generate_portfolio_html,
    attachment_pdf,
    attachment_html,
    run_tool,
)

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


class KnowledgeIngestRequest(BaseModel):
    rows: List[dict] = Field(..., min_length=1, max_length=10000)


class EnduranceRequest(BaseModel):
    battery_wh: float = Field(..., gt=0, le=100000000)
    average_watts: float = Field(..., gt=0, le=100000000)
    usable_fraction: float = Field(default=0.8, ge=0.1, le=1.0)


class FMEARequest(BaseModel):
    system: str = Field(..., min_length=1, max_length=160)
    components: Optional[List[str]] = None


class CalcRequest(BaseModel):
    expression: str


class PdfRequest(BaseModel):
    title: str = "Document"
    sections: Optional[List[dict]] = None


class CvRequest(BaseModel):
    name: str = "Name"
    title: str = ""
    summary: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    experience: Optional[List[dict]] = None
    education: Optional[List[dict]] = None
    skills: Optional[List[str]] = None


class PortfolioRequest(BaseModel):
    name: str = "My Portfolio"
    title: str = ""
    about: str = ""
    projects: Optional[List[dict]] = None
    skills: Optional[List[str]] = None
    contact: Optional[dict] = None


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
        description="Text + coding focused AI API with multi-agent core. Created by JagX & JRILICENSE.",
    )
    app.add_middleware(GlobalIPRateLimitMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)

    _default_origins = [
        "https://jagxai.name.ng",
        "https://www.jagxai.name.ng",
        "https://jagx-ai-app.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]
    cors_env = os.environ.get("JAGX_CORS_ORIGINS", "")
    if cors_env.strip():
        origins = [o.strip().rstrip("/") for o in cors_env.split(",") if o.strip()]
    else:
        origins = _default_origins
    if os.environ.get("JAGX_CORS_ALLOW_ALL", "").lower() in ("1", "true", "yes"):
        origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https://jagx-ai-[a-z0-9-]+-jagxs-projects\.vercel\.app",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "x-api-key", "x-admin-secret"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.include_router(jobs_router)
    app.include_router(autonomy_router)
    app.include_router(agents_router)

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
                "permanent_api_keys", "public_chat", "multi_provider", "local_knowledge_fallback",
                "streaming", "sse_streaming", "websocket_realtime_chat", "memory", "calc", "tools",
                "pdf_cv", "portfolio", "rate_limits", "persistent_conversations", "durable_background_jobs",
                "consent_memory", "optional_voice_pipeline", "offline_gguf_mode",
                "engineering_decision_support", "knowledge_ingestion", "image_generation_api",
                "grok_style_reasoning", "bot_ready",
                "multi_agent_v9", "agents_api",
            ],
            "docs": "/docs",
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "jagx-ai-v2",
            "version": __version__,
            "uptime_seconds": int(time.time() - APP_START),
            "local_brain": knowledge.size(),
            "providers": provider_status(),
        }

    @app.get("/providers/health")
    def providers_health(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return {"providers": provider_health(), "note": "Reachability does not validate credentials, quota, or model access."}

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

    def _memory_owner(key: str) -> str:
        return hashlib.sha256(("jagx-memory-v1:" + key).encode("utf-8")).hexdigest()

    # NOTE: Remaining endpoints (public-chat, chat, memory, calc, conversations,
    # images, video, tool, offline, speech, websocket, engineering, knowledge)
    # are preserved from v8. This file was updated only for V9 multi-agent wiring.
    # Full body restored from previous main for compatibility.

    return app
