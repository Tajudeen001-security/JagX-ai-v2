"""FastAPI application factory — JagX AI v8."""

from __future__ import annotations

import logging
import asyncio
import time
import hashlib
import hmac
import json
import os
import base64
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
        description="Text + coding focused AI API. Created by JagX & JRILICENSE.",
    )
    app.add_middleware(GlobalIPRateLimitMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.include_router(jobs_router)

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
                "permanent_api_keys", "multi_provider", "local_knowledge_fallback",
                "streaming", "sse_streaming", "websocket_realtime_chat", "memory", "calc", "tools",
                "pdf_cv", "portfolio", "rate_limits", "persistent_conversations", "durable_background_jobs" ,
                "consent_memory", "optional_voice_pipeline", "offline_gguf_mode",
                "engineering_decision_support", "knowledge_ingestion", "image_generation_api",
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

    @app.post("/video/generate")
    def generate_video(req: VideoGenerationRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        try:
            return video_generation.generate(req.prompt, req.duration, req.model, req.aspect_ratio)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.warning("video generation failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Video generation failed or is unavailable. Check provider key, access and quota.")

    @app.post("/images/generate")
    def generate_image(req: ImageGenerationRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        from jagx_api.image_generation import generate
        try:
            result = generate(req.prompt, req.width, req.height)
            return {"ok": True, **result}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/offline/status")
    def offline_status(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return {"offline_mode": settings.offline_mode, "local_model": local_model.status(), "local_knowledge_entries": knowledge.size(),
                "note": "Offline generation requires an installed llama-cpp-python runtime and a compatible local GGUF model file."}

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

    @app.get("/engineering/capabilities")
    def engineering_capabilities(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return engineering.capability_map()

    @app.post("/engineering/endurance")
    def engineering_endurance(req: EnduranceRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        try:
            return engineering.endurance_estimate(req.battery_wh, req.average_watts, req.usable_fraction)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/engineering/fmea-template")
    def engineering_fmea(req: FMEARequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return engineering.fmea_template(req.system, req.components)

    @app.get("/connected-accounts/providers")
    def connected_account_providers(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return {"providers":[{"id":"google","status":"not_configured"},{"id":"microsoft","status":"not_configured"},{"id":"github","status":"not_configured"}],
                "note":"OAuth is not connected until client credentials, callbacks, verified user identity, consent, and secure token storage are configured."}

    @app.get("/connected-accounts")
    def connected_accounts(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        return {"ok":True,"accounts":[],"note":"No connected accounts are configured."}

    @app.post("/knowledge/ingest")
    def ingest_knowledge(req: KnowledgeIngestRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key"),
                         x_admin_secret: Optional[str] = Header(None, alias="x-admin-secret")):
        _require_key(x_api_key)
        if not x_admin_secret or not hmac.compare_digest(x_admin_secret, settings.admin_secret):
            raise HTTPException(status_code=403, detail="Admin authorization required")
        rows=[]
        for row in req.rows:
            if not isinstance(row,dict):
                raise HTTPException(status_code=400,detail="Each knowledge row must be an object")
            q=str(row.get("question") or row.get("q") or "").strip()
            a=str(row.get("answer") or row.get("a") or "").strip()
            if not q or not a or len(q)>2000 or len(a)>12000:
                raise HTTPException(status_code=400,detail="Each row needs a question (max 2000 chars) and answer (max 12000 chars)")
            rows.append({"question":q,"answer":a})
        destination="knowledge_pack_ingested.json"
        temp=destination+".tmp"
        with open(temp,"w",encoding="utf-8") as handle:
            json.dump(rows,handle,ensure_ascii=False,indent=2)
        os.replace(temp,destination)
        count=knowledge.load_brain(".")
        return {"ok":True,"ingested":len(rows),"loaded_entries":count,"warning":"Ensure the host has persistent disk enabled; local files may disappear on redeploy."}

    @app.post("/speech/transcribe")
    async def speech_transcribe(audio: UploadFile = File(...), x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        payload=await audio.read(20*1024*1024+1)
        if len(payload)>20*1024*1024:
            raise HTTPException(status_code=413,detail="Audio exceeds 20 MB limit")
        try:
            return {"ok":True,**speech.transcribe(payload,audio.filename or "audio.wav")}
        except (RuntimeError,ValueError) as exc:
            raise HTTPException(status_code=503 if isinstance(exc,RuntimeError) else 400,detail=str(exc))

    @app.post("/speech/speak")
    async def speech_speak(text: str = Form(...), voice: Optional[str] = Form(None),
                           x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        _require_key(x_api_key)
        try:
            audio=await speech.synthesize(text,voice)
            return Response(content=audio,media_type="audio/mpeg",headers={"Cache-Control":"no-store"})
        except (RuntimeError,ValueError) as exc:
            raise HTTPException(status_code=503 if isinstance(exc,RuntimeError) else 400,detail=str(exc))

    @app.post("/voice/turn")
    async def voice_turn(audio: UploadFile = File(...), conversation_id: Optional[str] = Form(None),
                         training_consent: bool = Form(False),
                         x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key=_require_key(x_api_key)
        payload=await audio.read(20*1024*1024+1)
        if len(payload)>20*1024*1024:
            raise HTTPException(status_code=413,detail="Audio exceeds 20 MB limit")
        try:
            transcript=speech.transcribe(payload,audio.filename or "audio.wav")
            message=transcript["text"].strip()
            if not message:
                raise HTTPException(status_code=400,detail="No speech detected")
            history=[]
            if conversation_id:
                saved=conversation_store.get(key,conversation_id,limit=settings.max_history)
                if saved is None:
                    raise HTTPException(status_code=404,detail="Conversation not found")
                history=[{"role":m["role"],"content":m["content"]} for m in saved["messages"]]
                conversation_store.append(key,conversation_id,"user",message)
            memory_hits=memory_store.recall(_memory_owner(key),message)
            memory_context="\n".join(str(hit.get("text","")) for hit in memory_hits) if memory_hits else None
            result=generate_response(message,history,memory_context=memory_context,training_consent=training_consent)
            if conversation_id:
                conversation_store.append(key,conversation_id,"assistant",result.get("response",""),result.get("provider"))
            spoken=await speech.synthesize(result.get("response",""))
            return {"ok":True,"transcription":transcript,"response":result.get("response"),
                    "provider":result.get("provider"),"audio_base64":base64.b64encode(spoken).decode("ascii"),
                    "audio_mime_type":"audio/mpeg","conversation_id":conversation_id}
        except HTTPException:
            raise
        except (RuntimeError,ValueError) as exc:
            raise HTTPException(status_code=503 if isinstance(exc,RuntimeError) else 400,detail=str(exc))

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """Bidirectional real-time text chat; send {"message":"..."} JSON frames."""
        key = (websocket.headers.get("x-api-key") or "").strip()
        try:
            if not auth.is_valid_key(key):
                await websocket.close(code=4401, reason="Invalid or missing API key")
                return
            ok, info = auth.check_rate_limit(key)
            if not ok:
                await websocket.close(code=4429, reason=str(info)[:120])
                return
            await websocket.accept()
            await websocket.send_json({"type": "ready", "service": "JagX AI", "version": __version__})
            while True:
                payload = await websocket.receive_json()
                if not isinstance(payload, dict):
                    await websocket.send_json({"type": "error", "error": "Send a JSON object with a message field"})
                    continue
                msg = str(payload.get("message") or "").strip()
                if not msg or len(msg) > 32000:
                    await websocket.send_json({"type": "error", "error": "Message must contain 1–32000 characters"})
                    continue
                conversation_id = str(payload.get("conversation_id") or "").strip() or None
                history = payload.get("history") if isinstance(payload.get("history"), list) else None
                if conversation_id:
                    saved = conversation_store.get(key, conversation_id, limit=settings.max_history)
                    if saved is None:
                        await websocket.send_json({"type": "error", "error": "Conversation not found"})
                        continue
                    history = [{"role": m["role"], "content": m["content"]} for m in saved["messages"]]
                    conversation_store.append(key, conversation_id, "user", msg)
                memory_hits = memory_store.recall(_memory_owner(key), msg)
                memory_context = "\n".join(str(hit.get("text", "")) for hit in memory_hits) if memory_hits else None
                await websocket.send_json({"type": "start", "conversation_id": conversation_id})
                pieces = []
                try:
                    iterator = iter(stream_response(msg, history, memory_context))
                    sentinel = object()
                    while True:
                        piece = await asyncio.to_thread(next, iterator, sentinel)
                        if piece is sentinel:
                            break
                        if not piece:
                            continue
                        piece = str(piece)
                        pieces.append(piece)
                        await websocket.send_json({"type": "token", "token": piece})
                    answer = "".join(pieces)
                    if conversation_id and answer:
                        conversation_store.append(key, conversation_id, "assistant", answer)
                    await websocket.send_json({"type": "done", "conversation_id": conversation_id})
                except WebSocketDisconnect:
                    raise
                except Exception:
                    logger.exception("WebSocket chat turn failed")
                    await websocket.send_json({"type": "error", "error": "This response failed. Please retry."})
                    await websocket.send_json({"type": "done", "conversation_id": conversation_id})
        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected")
        except Exception:
            logger.exception("WebSocket connection failed")
            try:
                await websocket.close(code=1011, reason="Chat connection failed")
            except Exception:
                pass

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
        memory_hits = memory_store.recall(_memory_owner(key), msg)
        memory_context = "\n".join(str(hit.get("text", "")) for hit in memory_hits) if memory_hits else None
        if req.stream:
            fmt = (req.stream_format or "sse").lower()
            if fmt == "plain":
                def plain_gen():
                    chunks = []
                    try:
                        for piece in stream_response(msg, history, memory_context):
                            chunks.append(piece)
                            yield piece
                    finally:
                        if req.conversation_id and chunks:
                            conversation_store.append(key, req.conversation_id, "assistant", "".join(chunks))
                return StreamingResponse(
                    plain_gen(), media_type="text/plain; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
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
            return StreamingResponse(
                sse_gen(), media_type="text/event-stream; charset=utf-8",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        result = generate_response(msg, history, memory_context=memory_context, training_consent=req.training_consent)
        if req.conversation_id:
            conversation_store.append(key, req.conversation_id, "assistant", result.get("response", ""), result.get("provider"))
        info = auth.key_info(key)
        out = {
            "response": result["response"], "provider": result.get("provider"),
            "local_hit": result.get("local_hit", False),
            "rate_limit": f"tier={info.get('tier')}" if info else "ok", "version": __version__,
        }
        if result.get("attachment"):
            out["attachment"] = result["attachment"]
        return out

    @app.post("/memory")
    def memory_endpoint(req: MemoryRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
        key = _require_key(x_api_key)
        owner = _memory_owner(key)
        action = (req.action or "remember").lower()
        if action == "remember":
            if not req.consent:
                raise HTTPException(status_code=400, detail="Explicit consent is required to save personal memory")
            return memory_store.remember(owner, req.text, req.tags)
        if action == "recall":
            return {"ok": True, "hits": memory_store.recall(owner, req.query or req.text)}
        if action == "forget":
            return memory_store.forget(owner, req.memory_id)
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
