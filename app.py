"""
JagX AI 5.2
General Purpose AI + Multi-Image Vision
Created by JagX & JRILICENSE
"""

import os
import json
import secrets
import threading
import time
import re
import base64
from typing import Optional, List, Dict
from collections import defaultdict
from urllib.parse import quote
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ====================== CONFIG ======================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

KEYS_FILE = "keys.json"
KNOWLEDGE_FILE = "jagx_knowledge.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(
    k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip()
)

TIER_HOURLY_LIMITS = {
    "free": 60,
    "premium": 300,
    "premium_plus": 800,
    "master": None,
    "admin": None
}

app = FastAPI(
    title="JagX AI 5.2",
    description="General Purpose AI + Multi-Image Vision by JagX & JRILICENSE",
    version="5.2.0"
)

lock = threading.Lock()
rate_limit_store = defaultdict(list)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = f"""You are JagX AI 5.2 — a powerful general-purpose AI created by JagX and JRILICENSE.

STRICT IDENTITY RULES:
- Your name is JagX AI.
- You were created by JagX and JRILICENSE.
- Never say you were created by OpenAI, Alibaba, Qwen, Meta, Google, Anthropic, or any other company.
- Always introduce yourself as JagX AI by JagX & JRILICENSE when asked.

CAPABILITIES:
- Answer general knowledge questions
- Solve mathematics step by step
- Write and explain code in many languages
- Image generation and image understanding (vision)
- Internet search for real-time information
- Current date and time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

Be clear, helpful, accurate, and professional.
"""

# ====================== WATERMARK ======================
def add_invisible_watermark(text: str) -> str:
    ZWSP = "\u200B"
    ZWNJ = "\u200C"
    ZWJ  = "\u200D"
    pattern = [ZWNJ, ZWSP, ZWSP, ZWSP, ZWJ, ZWJ, ZWJ, ZWSP]
    watermark = "".join(pattern)

    if not text or len(text) < 15:
        return text + watermark

    third = len(text) // 3
    return (
        text[:3] + watermark +
        text[3:third] + watermark +
        text[third:third*2] + watermark +
        text[third*2:]
    )

# ====================== HELPERS ======================
def load_keys() -> dict:
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "w") as f:
            json.dump({}, f)
    with open(KEYS_FILE, "r") as f:
        return json.load(f)

def save_keys(keys: dict):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def load_knowledge() -> List[Dict]:
    if not os.path.exists(KNOWLEDGE_FILE):
        return []
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []

def search_knowledge(query: str) -> Optional[str]:
    knowledge = load_knowledge()
    query_lower = query.lower().strip()
    if not query_lower:
        return None
    for item in knowledge:
        q = item.get("question", "").lower()
        if q == query_lower or query_lower in q or q in query_lower:
            return item.get("answer")
    return None

def free_web_search(query: str) -> Optional[str]:
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return None
        texts = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        if not texts:
            texts = re.findall(r'class="result__snippet">(.*?)</td>', r.text, re.DOTALL)
        clean = []
        for t in texts[:4]:
            t = re.sub(r'<.*?>', '', t).strip()
            if len(t) > 40:
                clean.append(t)
        if clean:
            return "Here's what I found:\n\n" + "\n\n".join(clean)
        return None
    except:
        return None

def is_valid_key(key: str) -> bool:
    if not key:
        return False
    if key in PERMANENT_KEYS:
        return True
    keys = load_keys()
    return key in keys and keys[key].get("active", True)

def check_rate_limit(key: str) -> tuple:
    if key in PERMANENT_KEYS:
        return True, "unlimited"
    keys = load_keys()
    if key not in keys:
        return False, "Invalid API key"
    user = keys[key]
    if not user.get("active", True):
        return False, "This API key has been blocked"
    tier = user.get("tier", "free")
    limit = TIER_HOURLY_LIMITS.get(tier)
    if limit is None:
        return True, "unlimited"
    now = time.time()
    rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < 3600]
    if len(rate_limit_store[key]) >= limit:
        return False, f"Hourly limit reached ({limit} requests/hour)."
    rate_limit_store[key].append(now)
    remaining = limit - len(rate_limit_store[key])
    return True, f"{remaining} requests remaining this hour"

# ====================== LLM ======================
def call_external_llm(messages: list, max_tokens: int = 1500) -> Optional[str]:
    if HF_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.5
            }
            r = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except:
            pass
    if NVIDIA_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "meta/llama-3.1-8b-instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.5,
                "stream": False
            }
            r = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except:
            pass
    return None

def generate_response(user_message: str, history: Optional[List[Dict]] = None) -> str:
    local = search_knowledge(user_message)
    if local:
        return local
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for m in history[-10:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    external = call_external_llm(messages)
    if external:
        return external
    search_result = free_web_search(user_message)
    if search_result:
        return search_result
    return "I am JagX AI, created by JagX & JRILICENSE. I couldn't find a complete answer right now. Please try rephrasing your question."

# ====================== VISION ======================
def analyze_image_with_vision(images: List[str], question: str) -> str:
    if not images:
        return "No images were provided."

    cleaned_images = []
    for img in images:
        if "," in img:
            img = img.split(",")[1]
        cleaned_images.append(img)

    main_image = cleaned_images[0]

    if HF_TOKEN:
        try:
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "image": main_image,
                "question": question
            }
            r = requests.post(
                "https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base",
                headers=headers,
                json=payload,
                timeout=45
            )
            if r.status_code == 200:
                result = r.json()
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get("answer") or str(result[0])
                    if len(cleaned_images) > 1:
                        answer += f"\n\n(Note: You uploaded {len(cleaned_images)} images. I analyzed the first one.)"
                    return answer
                if isinstance(result, dict):
                    answer = result.get("answer") or result.get("generated_text") or str(result)
                    if len(cleaned_images) > 1:
                        answer += f"\n\n(Note: You uploaded {len(cleaned_images)} images. I analyzed the first one.)"
                    return answer
        except Exception as e:
            print("Vision error:", e)

    if len(cleaned_images) > 1:
        return (
            f"I successfully received {len(cleaned_images)} images. "
            "Full advanced multi-image vision is still being improved on JagX AI. "
            "Please describe the images or ask a specific question."
        )

    return (
        "I successfully received your image. "
        "Full advanced vision is still being improved on JagX AI. "
        "Please describe the image or ask a specific question."
    )

# ====================== MODELS ======================
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 1500
    history: Optional[List[Dict[str, str]]] = None

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024

class VisionRequest(BaseModel):
    images: List[str]
    question: str = "Describe the image(s) in detail."
    max_tokens: int = 1000

class CreateKeyRequest(BaseModel):
    owner_label: str
    admin_secret: str
    tier: str = "free"

class AdminKeyRequest(BaseModel):
    api_key: str
    admin_secret: str

class UpgradeKeyRequest(BaseModel):
    api_key: str
    new_tier: str
    admin_secret: str

class BlockKeyRequest(BaseModel):
    api_key: str
    active: bool
    admin_secret: str

# ====================== ROUTES ======================
@app.get("/")
def root():
    return {
        "status": "JagX AI 5.2 is running",
        "version": "5.2.0",
        "created_by": "JagX & JRILICENSE",
        "features": [
            "general_purpose",
            "mathematics",
            "multi_language_coding",
            "image_generation",
            "multi_image_vision",
            "free_search",
            "invisible_watermark",
            "hourly_rate_limit",
            "key_management"
        ]
    }

@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    allowed, quota_msg = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_msg)
    reply = generate_response(req.message, req.history)
    reply = add_invisible_watermark(reply)
    return {
        "response": reply,
        "model": "JagX AI 5.2",
        "quota": quota_msg
    }

@app.post("/image")
def generate_image(req: ImageRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    allowed, quota_msg = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_msg)
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width={req.width}&height={req.height}&nologo=true&model=flux"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            img_b64 = base64.b64encode(r.content).decode()
            return {
                "success": True,
                "source": "pollinations",
                "image_base64": img_b64,
                "format": "png",
                "quota": quota_msg
            }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {str(e)}")
    raise HTTPException(status_code=502, detail="Image generation failed")

@app.post("/vision")
def vision(req: VisionRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    allowed, quota_msg = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_msg)
    if not req.images or len(req.images) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required")
    answer = analyze_image_with_vision(req.images, req.question)
    answer = add_invisible_watermark(answer)
    return {
        "response": answer,
        "model": "JagX AI Vision",
        "images_received": len(req.images),
        "quota": quota_msg
    }

@app.post("/create-key")
def create_key(req: CreateKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    tier = req.tier.lower()
    if tier not in TIER_HOURLY_LIMITS:
        tier = "free"
    with lock:
        keys = load_keys()
        new_key = "jagx-" + secrets.token_hex(16)
        keys[new_key] = {
            "owner": req.owner_label,
            "active": True,
            "tier": tier,
            "created_at": time.time()
        }
        save_keys(keys)
    return {
        "api_key": new_key,
        "owner": req.owner_label,
        "tier": tier,
        "hourly_limit": TIER_HOURLY_LIMITS.get(tier)
    }

@app.get("/admin/keys")
def list_keys(admin_secret: str = Header(...)):
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    keys = load_keys()
    result = []
    for k, v in keys.items():
        result.append({
            "api_key": k,
            "owner": v.get("owner"),
            "tier": v.get("tier", "free"),
            "active": v.get("active", True),
            "created_at": v.get("created_at")
        })
    return {"total": len(result), "keys": result}

@app.post("/admin/block-key")
def block_key(req: BlockKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    with lock:
        keys = load_keys()
        if req.api_key not in keys:
            raise HTTPException(status_code=404, detail="Key not found")
        keys[req.api_key]["active"] = req.active
        save_keys(keys)
    return {"success": True, "message": "Key updated successfully"}

@app.post("/admin/delete-key")
def delete_key(req: AdminKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    with lock:
        keys = load_keys()
        if req.api_key not in keys:
            raise HTTPException(status_code=404, detail="Key not found")
        del keys[req.api_key]
        save_keys(keys)
    return {"success": True, "message": "Key deleted successfully"}

@app.post("/admin/upgrade-key")
def upgrade_key(req: UpgradeKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    new_tier = req.new_tier.lower()
    if new_tier not in TIER_HOURLY_LIMITS:
        raise HTTPException(status_code=400, detail="Invalid tier")
    with lock:
        keys = load_keys()
        if req.api_key not in keys:
            raise HTTPException(status_code=404, detail="Key not found")
        keys[req.api_key]["tier"] = new_tier
        save_keys(keys)
    return {
        "success": True,
        "message": f"Key upgraded to {new_tier}",
        "hourly_limit": TIER_HOURLY_LIMITS[new_tier]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)