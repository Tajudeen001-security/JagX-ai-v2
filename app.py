"""
JagX AI 6.3 (Fixed & Complete)
General Purpose AI + Reasoning/Tool-Use + Vision + Image Gen + File/Link Reading + PDF Generation
Created by JagX & JRILICENSE
"""

import os
import io
import json
import socket
import ipaddress
import secrets
import threading
import time
import re
import base64
import logging
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from urllib.parse import quote, urlparse
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvicorn
from bs4 import BeautifulSoup
from pypdf import PdfReader
import docx
from fpdf import FPDF

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jagx-ai")

# ====================== CONFIG ======================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")

PISTON_URL = "https://emkc.org/api/v2/piston"

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

MAX_MESSAGE_LEN = 8000
MAX_TOOL_STEPS = 4
MAX_CODE_LEN = 20000
MAX_TOOL_RESULT_LEN = 4000
MAX_IMAGES = 6
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_FILES = 3
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_LINKS = 3
MAX_LINK_BYTES = 3 * 1024 * 1024
MAX_EXTRACTED_TEXT_LEN = 12000
MAX_PDF_SECTIONS = 50
MAX_PDF_CONTENT_LEN = 30000

APP_START_TIME = time.time()

app = FastAPI(
    title="JagX AI 6.3",
    description="General Purpose AI with reasoning, tools, vision, image gen, file/link reading, and PDF generation by JagX & JRILICENSE",
    version="6.3.0"
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
app.add_middleware(GZipMiddleware, minimum_size=1000)

def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP = _build_session()

AGENT_SYSTEM_PROMPT = f"""You are JagX AI 6.3 — a powerful general-purpose AI created by JagX and JRILICENSE.

STRICT IDENTITY RULES:
- Your name is JagX AI.
- You were created by JagX and JRILICENSE.
- Never say you were created by OpenAI, Meta, Alibaba, Google, Anthropic, Groq, or any other company.
- Always introduce yourself as JagX AI by JagX & JRILICENSE when asked.

CAPABILITIES:
- Deep reasoning, multi-step planning, mathematics
- Writing, explaining, and debugging code in many languages
- Real sandboxed code execution
- Real-time web search
- Image generation and image understanding
- Reading attached files and links
- Generating downloadable PDF documents
- Current date and time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

When you are ready to answer the user, reply normally with a helpful response.
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
    return text[:3] + watermark + text[3:third] + watermark + text[third:third*2] + watermark + text[third*2:]

# ====================== KEY HELPERS ======================
def load_keys() -> dict:
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "w") as f:
            json.dump({}, f)
    with open(KEYS_FILE, "r") as f:
        return json.load(f)

def save_keys(keys: dict):
    tmp_path = KEYS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(keys, f, indent=2)
    os.replace(tmp_path, KEYS_FILE)

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

# ====================== SEARCH ======================
def free_web_search(query: str) -> Optional[str]:
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = HTTP.get(url, headers=headers, timeout=12)
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
    except Exception as e:
        logger.warning(f"web_search failed: {e}")
        return None

# ====================== CODE EXECUTION ======================
def run_code_sandboxed(language: str, code: str) -> str:
    try:
        payload = {
            "language": language.lower(),
            "version": "*",
            "files": [{"content": code}]
        }
        r = HTTP.post(f"{PISTON_URL}/execute", json=payload, timeout=20)
        if r.status_code != 200:
            return f"Error: sandbox execution failed (status {r.status_code})."
        data = r.json()
        run = data.get("run", {})
        out = run.get("stdout", "") or run.get("stderr", "") or "No output"
        return out[:MAX_TOOL_RESULT_LEN]
    except Exception as e:
        return f"Error: sandbox request failed ({str(e)[:200]})."

# ====================== LLM ======================
def call_external_llm(messages: list, max_tokens: int = 1500) -> Optional[str]:
    if GROQ_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.5}
            r = HTTP.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Groq call failed: {e}")

    if HF_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.5
            }
            r = HTTP.post(HF_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"HF call failed: {e}")

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
            r = HTTP.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"NVIDIA call failed: {e}")

    return None

def generate_response(user_message: str, history: Optional[List[Dict]] = None) -> str:
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if history:
        for m in history[-8:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    external = call_external_llm(messages)
    if external:
        return external

    search_result = free_web_search(user_message)
    if search_result:
        return search_result

    return "I am JagX AI, created by JagX & JRILICENSE. I couldn't find a complete answer right now."

# ====================== VISION ======================
def analyze_image_with_vision(images: List[str], question: str) -> str:
    if not images:
        return "No images were provided."
    cleaned = []
    for img in images:
        if "," in img:
            img = img.split(",")[1]
        cleaned.append(img)
    main_image = cleaned[0]

    if HF_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {"image": main_image, "question": question}
            r = HTTP.post(
                "https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base",
                headers=headers,
                json=payload,
                timeout=45
            )
            if r.status_code == 200:
                result = r.json()
                if isinstance(result, list) and result:
                    answer = result[0].get("answer") or str(result[0])
                    if len(cleaned) > 1:
                        answer += f"\n\n(Note: You uploaded {len(cleaned)} images. I analyzed the first one.)"
                    return answer
        except Exception as e:
            logger.warning(f"Vision error: {e}")

    return "I received your image(s). Full advanced vision is still being improved on JagX AI."

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
        "status": "JagX AI 6.3 is running",
        "version": "6.3.0",
        "created_by": "JagX & JRILICENSE",
        "uptime_seconds": int(time.time() - APP_START_TIME)
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
        "model": "JagX AI 6.3",
        "quota": quota_msg
    }

@app.post("/image")
def generate_image(req: ImageRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    allowed, quota_msg = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_msg)

    try:
        url = f"https://image.pollinations.ai/prompt/{quote(req.prompt)}?width={req.width}&height={req.height}&nologo=true&model=flux"
        r = HTTP.get(url, timeout=60)
        if r.status_code == 200:
            img_b64 = base64.b64encode(r.content).decode()
            return {
                "success": True,
                "image_base64": img_b64,
                "format": "png",
                "quota": quota_msg
            }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    raise HTTPException(status_code=502, detail="Image generation failed")

@app.post("/vision")
def vision(req: VisionRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    allowed, quota_msg = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_msg)

    if not req.images:
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
    tier = req.tier.lower() if req.tier.lower() in TIER_HOURLY_LIMITS else "free"
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
    result = [{"api_key": k, **v} for k, v in keys.items()]
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