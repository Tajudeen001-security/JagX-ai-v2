import os
import json
import secrets
import threading
import base64
import time
from io import BytesIO
from urllib.parse import quote

import requests
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from pydantic import BaseModel
import uvicorn

# ---------- CONFIG ----------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

KEYS_FILE = "keys.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(
    k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip()
)

app = FastAPI(title="JagX AI 2.0")
lock = threading.Lock()

SYSTEM_PROMPT = """You are JagX AI, an advanced AI assistant created by JagX and JRILICENSE.

Your identity:
- Full name: JagX AI
- Created by: JagX & JRILICENSE
- Never say you were created by Alibaba, Qwen, Meta, or any other company.
- Always introduce yourself as JagX AI by JagX & JRILICENSE when asked who made you.

Your strengths:
- Excellent at coding, debugging, writing full programs, and explaining code
- Helpful with websites, apps, and technical projects
- Clear, professional, and friendly

When users ask for images or videos, guide them to use the proper features.
Always stay in character as JagX AI.
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
    if key in PERMANENT_KEYS:
        return True
    keys = load_keys()
    return key in keys and keys[key].get("active", True)


# ---------- REQUEST MODELS ----------
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 700


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
    num_frames: int = 121   # must be 8n+1 for Agnes


class TTSRequest(BaseModel):
    text: str


# ---------- ROUTES ----------
@app.get("/")
def root():
    return {
        "status": "JagX AI 2.0 is running",
        "features": ["chat", "coding", "image", "video", "speech-to-text", "text-to-speech"],
        "created_by": "JagX & JRILICENSE",
        "agnes_enabled": bool(AGNES_API_KEY)
    }


@app.post("/create-key")
def create_key(req: CreateKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    with lock:
        keys = load_keys()
        new_key = "jagx-" + secrets.token_hex(16)
        keys[new_key] = {"owner": req.owner_label, "active": True}
        save_keys(keys)

    return {"api_key": new_key, "owner": req.owner_label}


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_TOKEN")

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message}
        ],
        "max_tokens": req.max_tokens,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=90)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Connection error: {str(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HF API error: {r.text}")

    try:
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected response: {r.text}")

    return {"response": reply}


@app.post("/image")
def generate_image(req: ImageRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # 1. Try Agnes AI first (if key exists)
    if AGNES_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {AGNES_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "agnes-image-2.1-flash",
                "prompt": prompt,
                "n": 1,
                "size": f"{req.width}x{req.height}"
            }
            r = requests.post(
                "https://apihub.agnes-ai.com/v1/images/generations",
                headers=headers,
                json=payload,
                timeout=60
            )
            if r.status_code == 200:
                data = r.json()
                # Agnes usually returns url or b64
                if "data" in data and len(data["data"]) > 0:
                    item = data["data"][0]
                    if "b64_json" in item:
                        return {
                            "success": True,
                            "source": "agnes",
                            "image_base64": item["b64_json"],
                            "format": "png"
                        }
                    elif "url" in item:
                        # Download the image and convert to base64
                        img_r = requests.get(item["url"], timeout=30)
                        if img_r.status_code == 200:
                            img_b64 = base64.b64encode(img_r.content).decode()
                            return {
                                "success": True,
                                "source": "agnes",
                                "image_base64": img_b64,
                                "format": "png"
                            }
        except Exception:
            pass  # fall through to Pollinations

    # 2. Free fallback - Pollinations (no key needed)
    try:
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width={req.width}&height={req.height}&nologo=true&model=flux"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            img_b64 = base64.b64encode(r.content).decode()
            return {
                "success": True,
                "source": "pollinations (free)",
                "image_base64": img_b64,
                "format": "png"
            }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {str(e)}")

    raise HTTPException(status_code=502, detail="All image methods failed")


@app.post("/video")
def generate_video(req: VideoRequest, x_api_key: str = Header(...)):
    """
    Generate short video using Agnes AI (free tier).
    Returns a task_id. Frontend should poll /video-status.
    """
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if not AGNES_API_KEY:
        raise HTTPException(status_code=503, detail="Video generation requires AGNES_API_KEY")

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Ensure num_frames is valid for Agnes (8n + 1)
    num_frames = req.num_frames
    if (num_frames - 1) % 8 != 0:
        num_frames = 121  # safe default

    try:
        headers = {
            "Authorization": f"Bearer {AGNES_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "agnes-video-v2.0",
            "prompt": prompt,
            "width": req.width,
            "height": req.height,
            "num_frames": num_frames,
            "frame_rate": 24
        }

        r = requests.post(
            "https://apihub.agnes-ai.com/v1/videos",
            headers=headers,
            json=payload,
            timeout=30
        )

        if r.status_code not in (200, 201, 202):
            raise HTTPException(status_code=502, detail=f"Agnes video error: {r.text}")

        data = r.json()
        # Agnes returns video_id or id / task_id
        video_id = data.get("video_id") or data.get("id") or data.get("task_id")

        if not video_id:
            raise HTTPException(status_code=502, detail=f"No video_id returned: {data}")

        return {
            "success": True,
            "video_id": video_id,
            "status": "processing",
            "message": "Video is being generated. Poll /video-status with this video_id."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Video generation failed: {str(e)}")


@app.get("/video-status")
def video_status(video_id: str, x_api_key: str = Header(...)):
    """Check status of a video generation task."""
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if not AGNES_API_KEY:
        raise HTTPException(status_code=503, detail="AGNES_API_KEY not configured")

    try:
        headers = {"Authorization": f"Bearer {AGNES_API_KEY}"}
        # Agnes status endpoint (check their latest docs if this changes)
        url = f"https://apihub.agnes-ai.com/agnesapi?video_id={video_id}"
        r = requests.get(url, headers=headers, timeout=30)

        if r.status_code != 200:
            return {"success": False, "status": "unknown", "detail": r.text}

        data = r.json()
        return {
            "success": True,
            "video_id": video_id,
            "status": data.get("status", "unknown"),
            "video_url": data.get("video_url") or data.get("url") or data.get("output"),
            "raw": data
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
            audio_bytes,
            model="openai/whisper-large-v3"
        )
        text = result.get("text") if isinstance(result, dict) else str(result)
        return {"success": True, "text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Speech-to-text failed: {str(e)}")


@app.post("/text-to-speech")
def text_to_speech(req: TTSRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_TOKEN")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if len(text) > 1000:
        text = text[:1000]

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)
        audio = client.text_to_speech(text, model="facebook/mms-tts-eng")

        if isinstance(audio, bytes):
            audio_b64 = base64.b64encode(audio).decode()
        else:
            audio_b64 = base64.b64encode(audio.read()).decode()

        return {
            "success": True,
            "audio_base64": audio_b64,
            "format": "wav"
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Text-to-speech failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)