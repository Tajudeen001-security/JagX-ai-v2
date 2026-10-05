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
from jagx_api.chat import generate_response, stream_response, stream_sse
from jagx_api.config import settings
from jagx_api.providers import provider_status
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


def create_app() -> FastAPI:
    app = FastAPI(
        title="JagX AI",
        version=__version__,
        description="Text + coding focused AI API. Created by JagX & JRILICENSE.",
    )
    app.add_middleware(GlobalIPRateLimitMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
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
                "permanent_api_keys", "multi_provider", "local_knowledge_fallback",
                "streaming", "sse_streaming", "memory", "calc", "tools",
                "pdf_cv", "portfolio", "rate_limits",
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
    def chat(req: ChatRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = _require_key(x_api_key)
        msg = req.message.strip()
        if not msg:
            raise HTTPException(status_code=400, detail="Empty message")
        if req.stream:
            fmt = (req.stream_format or "sse").lower()
            if fmt == "plain":
                def plain_gen():
                    for piece in stream_response(msg, req.history):
                        yield piece
                return StreamingResponse(
                    plain_gen(),
                    media_type="text/plain; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            def sse_gen():
                for event in stream_sse(msg, req.history):
                    yield event
            return StreamingResponse(
                sse_gen(),
                media_type="text/event-stream; charset=utf-8",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        result = generate_response(msg, req.history)
        info = auth.key_info(key)
        out = {
            "response": result["response"],
            "provider": result.get("provider"),
            "local_hit": result.get("local_hit", False),
            "rate_limit": f"tier={info.get('tier')}" if info else "ok",
            "version": __version__,
        }
        if result.get("attachment"):
            out["attachment"] = result["attachment"]
        return out

    @app.post("/memory")
    def memory_endpoint(req: MemoryRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        action = (req.action or "remember").lower()
        if action == "remember":
            return memory_store.remember(req.user_id, req.text, req.tags)
        if action == "recall":
            return {"ok": True, "hits": memory_store.recall(req.user_id, req.query or req.text)}
        if action == "forget":
            return memory_store.forget(req.user_id, req.memory_id)
        raise HTTPException(status_code=400, detail=f"unknown action: {action}")

    @app.post("/calc")
    def calc(req: CalcRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return safe_calc(req.expression)

    @app.post("/pdf")
    def make_pdf(req: PdfRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        try:
            return attachment_pdf("document.pdf", generate_pdf_document(req.title, req.sections or []))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/cv")
    def make_cv(req: CvRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        try:
            payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            return attachment_pdf("cv.pdf", generate_cv_pdf(payload))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/portfolio")
    def make_portfolio(req: PortfolioRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        try:
            payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            return attachment_html("portfolio.html", generate_portfolio_html(payload))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/tool")
    def tool_endpoint(req: ToolRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        text, attachment = run_tool(req.tool, req.input or {})
        out = {"ok": True, "result": text}
        if attachment:
            out["attachment"] = attachment
        return out

    return app


app = create_app()
