import os
import json
import secrets
import threading
import base64
from io import BytesIO

import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# ---------- CONFIG ----------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# Better model for coding + general intelligence
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"          # Good balance of coding + reasoning
# Alternative strong options: "Qwen/Qwen2.5-Coder-7B-Instruct" or "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
- Never say you were created by Alibaba, Qwen, or any other company.
- Always introduce yourself as JagX AI by JagX & JRILICENSE when asked who made you.

Your capabilities:
- Excellent at coding, debugging, explaining code, and writing complete programs
- Helpful with website building, app ideas, and technical guidance
- Creative and clear in explanations
- Professional, friendly, and concise

When the user asks for images, tell them you can generate images and ask them to use the image feature.
Always stay in character as JagX AI.
"""


# ---------- REQUEST MODELS ----------
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 600


class CreateKeyRequest(BaseModel):
    owner_label: str
    admin_secret: str


class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"status": "JagX AI 2.0 is running", "features": ["chat", "coding", "image"]}


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
        raise HTTPException(status_code=500, detail="Server misconfigured: missing HF_TOKEN")

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
    except requests.exceptions.RequestException as e:
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
    """
    Generate image.
    First tries Hugging Face Inference Providers.
    Falls back to free Pollinations if HF fails.
    """
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # --- Method 1: Try Hugging Face (better quality when credits available) ---
    if HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=HF_TOKEN)
            # Using a fast model that often has free availability
            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-schnell",  # Fast + good quality
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
        except Exception as e:
            # If HF fails (no credits / model not available), fall to free method
            pass

    # --- Method 2: Free fallback - Pollinations.ai (no key needed) ---
    try:
        # Pollinations is completely free (rate limited)
        pollinations_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width={req.width}&height={req.height}&nologo=true"
        r = requests.get(pollinations_url, timeout=60)
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

    raise HTTPException(status_code=502, detail="All image generation methods failed")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
