"""
JagX AI 3.0 — Strong Coding Backend
Upgrades over v2:
- Stronger default model (Qwen2.5-Coder / larger instruct)
- Multi-turn chat history
- Higher max_tokens
- OpenAI-compatible /v1/chat/completions
- Coding-optimized system prompt
- Temperature + model selection
- Better errors + retries
"""

import os
import json
import secrets
import threading
import base64
import time
from urllib.parse import quote
from typing import List, Optional, Any

import requests
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ---------- CONFIG ----------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

DEFAULT_MODEL = os.environ.get("CHAT_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "Qwen/Qwen2.5-14B-Instruct")
# Optional stronger models:
# Qwen/Qwen2.5-Coder-32B-Instruct
# Qwen/Qwen2.5-72B-Instruct

KEYS_FILE = "keys.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(
    k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip()
)

app = FastAPI(
    title="JagX AI 3.0",
    description="Strong coding + multimodal AI API by JagX & JRILICENSE",
    version="3.0.0",
)
lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are JagX AI 3.0 — an elite AI engineer and cybersecurity specialist created by JagX & JRILICENSE.

IDENTITY (never break character):
- Full name: JagX AI
- Created by: JagX & JRILICENSE
- Never claim you were made by Alibaba, Qwen, Meta, OpenAI, Google, or any other company.
- When asked who made you, always say: JagX AI by JagX & JRILICENSE.

CORE STRENGTHS:
- Expert software engineer: production-quality code, full-stack apps, clean architecture
- Cybersecurity: secure coding, OWASP, auth/JWT, input validation, hardening, defensive tools
- Debugging, refactoring, testing, documentation
- Clear step-by-step reasoning before writing code
- Never produce empty files or placeholder-only code

CODING RULES:
1. Think before you code. Prefer small correct steps.
2. Write complete, runnable code — never empty files.
3. For full-stack: structure → backend → models → frontend → auth → README.
4. Include error handling and basic security by default.
5. Explain how to run the project when you finish.

Be professional, precise, and powerful.
"""


# ---------- KEY STORAGE ----------
def load_keys():
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "w") as f:
            json.dump({}, f)
    with open(KEYS_FILE, "r") as f:
        return json.load(f)


def save_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


def is_valid_key(key: str) -> bool:
    if not key:
        return False
    if key in PERMANENT_KEYS:
        return True
    keys = load_keys()
    return key in keys and keys[key].get("active", True)


# ---------- REQUEST MODELS ----------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 2000
    temperature: float = 0.3
    model: Optional[str] = None
    history: Optional[List[ChatMessage]] = None
    system: Optional[str] = None


class CreateKeyRequest(BaseModel):
    owner_label: str
    admin_secret: str


class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


class VideoRequest(BaseModel):
    prompt: str
    width: int = 1152
    height: int = 768
    num_frames: int = 121


class TTSRequest(BaseModel):
    text: str


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False


# ---------- HELPERS ----------
def call_hf_chat(
    messages: list,
    max_tokens: int = 2000,
    temperature: float = 0.3,
    model: Optional[str] = None,
) -> str:
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Server missing HF_TOKEN")

    use_model = model or DEFAULT_MODEL
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": use_model,
        "messages": messages,
        "max_tokens": min(max_tokens, 4096),
        "temperature": temperature,
    }

    last_err = None
    for attempt_model in [use_model, FALLBACK_MODEL]:
        payload["model"] = attempt_model
        try:
            r = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            last_err = f"{attempt_model}: {r.status_code} {r.text[:300]}"
        except Exception as e:
            last_err = str(e)
            continue

    raise HTTPException(status_code=502, detail=f"HF API failed: {last_err}")
# ---------- ROUTES ----------
@app.get("/")
def root():
    return {
        "status": "JagX AI 3.0 is running",
        "version": "3.0.0",
        "model": DEFAULT_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "features": [
            "chat", "multi-turn", "coding", "openai-compatible",
            "image", "video", "speech-to-text", "text-to-speech",
        ],
        "created_by": "JagX & JRILICENSE",
        "agnes_enabled": bool(AGNES_API_KEY),
    }


@app.post("/create-key")
def create_key(req: CreateKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    with lock:
        keys = load_keys()
        new_key = "jagx-" + secrets.token_hex(16)
        keys[new_key] = {"owner": req.owner_label, "active": True, "created": time.time()}
        save_keys(keys)
    return {"api_key": new_key, "owner": req.owner_label}


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    system = req.system or SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]

    if req.history:
        for m in req.history[-20:]:
            if m.role in ("user", "assistant", "system") and m.content:
                messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": req.message})

    reply = call_hf_chat(
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        model=req.model,
    )
    return {
        "response": reply,
        "model": req.model or DEFAULT_MODEL,
        "version": "3.0",
    }


@app.post("/v1/chat/completions")
def openai_compatible(
    req: OpenAIChatRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    key = x_api_key
    if not key and authorization:
        key = authorization.replace("Bearer ", "").strip()
    if not is_valid_key(key or ""):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    reply = call_hf_chat(
        messages=messages,
        max_tokens=req.max_tokens or 2000,
        temperature=req.temperature if req.temperature is not None else 0.3,
        model=req.model,
    )

    return {
        "id": f"jagx-{secrets.token_hex(8)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or DEFAULT_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/image")
def generate_image(req: ImageRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if AGNES_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "agnes-image-2.1-flash",
                "prompt": prompt,
                "n": 1,
                "size": f"{req.width}x{req.height}",
            }
            r = requests.post(
                "https://apihub.agnes-ai.com/v1/images/generations",
                headers=headers, json=payload, timeout=60,
            )
            if r.status_code == 200:
                data = r.json()
                if "data" in data and data["data"]:
                    item = data["data"][0]
                    if "b64_json" in item:
                        return {
                            "success": True,
                            "source": "agnes",
                            "image_base64": item["b64_json"],
                            "format": "png",
                        }
                    if "url" in item:
                        img_r = requests.get(item["url"], timeout=30)
                        if img_r.status_code == 200:
                            return {
                                "success": True,
                                "source": "agnes",
                                "image_base64": base64.b64encode(img_r.content).decode(),
                                "format": "png",
                            }
        except Exception:
            pass

    try:
        url = (
            f"https://image.pollinations.ai/prompt/{quote(prompt)}"
            f"?width={req.width}&height={req.height}&nologo=true&model=flux"
        )
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return {
                "success": True,
                "source": "pollinations (free)",
                "image_base64": base64.b64encode(r.content).decode(),
                "format": "png",
            }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {e}")

    raise HTTPException(status_code=502, detail="All image methods failed")


@app.post("/video")
def generate_video(req: VideoRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if not AGNES_API_KEY:
        raise HTTPException(status_code=503, detail="Video requires AGNES_API_KEY")
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    num_frames = req.num_frames if (req.num_frames - 1) % 8 == 0 else 121
    try:
        headers = {"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "agnes-video-v2.0",
            "prompt": prompt,
            "width": req.width,
            "height": req.height,
            "num_frames": num_frames,
            "frame_rate": 24,
        }
        r = requests.post(
            "https://apihub.agnes-ai.com/v1/videos",
            headers=headers, json=payload, timeout=30,
        )
        if r.status_code not in (200, 201, 202):
            raise HTTPException(status_code=502, detail=f"Agnes video error: {r.text}")
        data = r.json()
        video_id = data.get("video_id") or data.get("id") or data.get("task_id")
        if not video_id:
            raise HTTPException(status_code=502, detail=f"No video_id: {data}")
        return {
            "success": True,
            "video_id": video_id,
            "status": "processing",
            "message": "Poll /video-status with this video_id",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/video-status")
def video_status(video_id: str, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if not AGNES_API_KEY:
        raise HTTPException(status_code=503, detail="AGNES_API_KEY not configured")
    try:
        headers = {"Authorization": f"Bearer {AGNES_API_KEY}"}
        r = requests.get(
            f"https://apihub.agnes-ai.com/agnesapi?video_id={video_id}",
            headers=headers, timeout=30,
        )
        if r.status_code != 200:
            return {"success": False, "status": "unknown", "detail": r.text}
        data = r.json()
        return {
            "success": True,
            "video_id": video_id,
            "status": data.get("status", "unknown"),
            "video_url": data.get("video_url") or data.get("url") or data.get("output"),
            "raw": data,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/speech-to-text")
async def speech_to_text(x_api_key: str = Header(...), file: UploadFile = File(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_TOKEN")
    try:
        audio_bytes = await file.read()
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)
        result = client.automatic_speech_recognition(
            audio_bytes, model="openai/whisper-large-v3"
        )
        text = result.get("text") if isinstance(result, dict) else str(result)
        return {"success": True, "text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"STT failed: {e}")


@app.post("/text-to-speech")
def text_to_speech(req: TTSRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_TOKEN")
    text = req.text.strip()[:1000]
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)
        audio = client.text_to_speech(text, model="facebook/mms-tts-eng")
        if isinstance(audio, bytes):
            audio_b64 = base64.b64encode(audio).decode()
        else:
            audio_b64 = base64.b64encode(audio.read()).decode()
        return {"success": True, "audio_base64": audio_b64, "format": "wav"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)