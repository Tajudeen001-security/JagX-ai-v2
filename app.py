import os
import json
import secrets
import threading
import base64
from io import BytesIO

import requests
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ---------- CONFIG ----------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# Strong model for coding + general use
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

KEYS_FILE = "keys.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(
    k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip()
)

app = FastAPI(title="JagX AI 2.0")
lock = threading.Lock()


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


# ---------- SYSTEM PROMPT ----------
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

When users ask for images or voice features, guide them to use the proper endpoints.
Always stay in character as JagX AI.
"""


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


class TTSRequest(BaseModel):
    text: str
    voice: str = "default"   # reserved for future voices


# ---------- ROUTES ----------
@app.get("/")
def root():
    return {
        "status": "JagX AI 2.0 is running",
        "features": ["chat", "coding", "image", "speech-to-text", "text-to-speech"],
        "created_by": "JagX & JRILICENSE"
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

    # Try Hugging Face first
    if HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=HF_TOKEN)
            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-schnell"
            )
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {
                "success": True,
                "source": "huggingface",
                "image_base64": img_str,
                "format": "png"
            }
        except Exception:
            pass

    # Free fallback - Pollinations
    try:
        from urllib.parse import quote
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width={req.width}&height={req.height}&nologo=true"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            img_str = base64.b64encode(r.content).decode()
            return {
                "success": True,
                "source": "pollinations (free)",
                "image_base64": img_str,
                "format": "png"
            }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {str(e)}")

    raise HTTPException(status_code=502, detail="All image methods failed")


@app.post("/speech-to-text")
async def speech_to_text(
    x_api_key: str = Header(...),
    file: UploadFile = File(...)
):
    """
    Convert uploaded audio → text
    Accepts: wav, mp3, m4a, webm, ogg
    """
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_TOKEN")

    try:
        audio_bytes = await file.read()

        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)

        # Whisper is the most reliable free STT model
        result = client.automatic_speech_recognition(
            audio_bytes,
            model="openai/whisper-large-v3"
        )

        text = result.get("text") if isinstance(result, dict) else str(result)

        return {
            "success": True,
            "text": text.strip(),
            "model": "whisper-large-v3"
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Speech-to-text failed: {str(e)}")


@app.post("/text-to-speech")
def text_to_speech(req: TTSRequest, x_api_key: str = Header(...)):
    """
    Convert text → speech audio (base64)
    """
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_TOKEN")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 characters)")

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)

        # Using a widely available TTS model
        audio = client.text_to_speech(
            text,
            model="facebook/mms-tts-eng"   # English TTS
        )

        # audio is usually bytes
        if isinstance(audio, bytes):
            audio_b64 = base64.b64encode(audio).decode()
        else:
            # Some versions return a file-like object
            audio_b64 = base64.b64encode(audio.read()).decode()

        return {
            "success": True,
            "audio_base64": audio_b64,
            "format": "wav",
            "model": "mms-tts-eng"
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Text-to-speech failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
