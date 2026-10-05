"""FastAPI application factory — JagX AI v8."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from jagx_api import __version__, auth, knowledge, memory_store
from jagx_api.chat import generate_response, stream_response
from jagx_api.config import settings
from jagx_api.providers import provider_status
from jagx_api.tools import safe_calc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jagx-ai")

APP_START = time.time()


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


class CalcRequest(BaseModel):
    expression: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="JagX AI",
        version=__version__,
        description="Text + coding focused AI API. Created by JagX & JRILICENSE.",
    )

    app.add_middleware(GlobalIPRateLimitMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.on_event("startup")
    def _startup() -> None:
        n = knowledge.load_brain(".")
        logger.info("JagX AI %s started — brain=%s providers=%s", __version__, n, provider_status())

    def _require_key(x_api_key: Optional[str]) -> str:
        key = (x_api_key or "").strip()
        if not auth.is_valid_key(key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        ok, info = auth.check_rate_limit(key)
        if not ok:
            raise HTTPException(status_code=429, detail=info)
        return key

    @app.get("/")
    def root():
        return {
            "status": f"JagX AI {__version__} is running",
            "version": __version__,
            "created_by": "JagX & JRILICENSE",
            "features": [
                "permanent_api_keys",
                "multi_provider",
                "local_knowledge_fallback",
                "streaming",
                "memory",
                "calc",
                "tools",
                "rate_limits",
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

    @app.post("/chat")
    def chat(
        req: ChatRequest,
        x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    ):
        key = _require_key(x_api_key)
        msg = req.message.strip()
        if not msg:
            raise HTTPException(status_code=400, detail="Empty message")

        if req.stream:
            def event_gen():
                for piece in stream_response(msg, req.history):
                    yield piece

            return StreamingResponse(event_gen(), media_type="text/plain; charset=utf-8")

        result = generate_response(msg, req.history)
        info = auth.key_info(key)
        return {
            "response": result["response"],
            "provider": result.get("provider"),
            "local_hit": result.get("local_hit", False),
            "rate_limit": f"tier={info.get('tier')}" if info else "ok",
            "version": __version__,
        }

    @app.post("/memory")
    def memory_endpoint(
        req: MemoryRequest,
        x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    ):
        _require_key(x_api_key)
        action = (req.action or "remember").lower()
        if action == "remember":
            return memory_store.remember(req.user_id, req.text, req.tags)
        if action == "recall":
            hits = memory_store.recall(req.user_id, req.query or req.text)
            return {"ok": True, "hits": hits}
        if action == "forget":
            return memory_store.forget(req.user_id, req.memory_id)
        raise HTTPException(status_code=400, detail=f"unknown action: {action}")

    @app.post("/calc")
    def calc(
        req: CalcRequest,
        x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    ):
        _require_key(x_api_key)
        return safe_calc(req.expression)

    return app


app = create_app()
