"""
JagX AI 4.2
Created by JagX & JRILICENSE

Features:
- Hourly Rate Limiting
- Full Key Management
- Local Knowledge Fallback
- Free Internet Search
- Invisible Watermark on every response
- External LLMs only when keys are available
"""

import os
import json
import secrets
import threading
import time
import re
from typing import Optional, List, Dict
from collections import defaultdict
from urllib.parse import quote

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

# Hourly limits
TIER_HOURLY_LIMITS = {
    "free": 60,
    "premium": 300,
    "premium_plus": 800,
    "master": None,
    "admin": None
}

app = FastAPI(
    title="JagX AI 4.2",
    description="Independent AI by JagX & JRILICENSE with Invisible Watermark",
    version="4.2.0"
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

SYSTEM_PROMPT = """You are JagX AI, an advanced independent AI created by JagX and JRILICENSE.

STRICT RULES:
- Your name is JagX AI.
- You were created by JagX and JRILICENSE.
- Never say you were created by OpenAI, Alibaba, Qwen, Meta, Google, or any other company.
- Always present yourself as JagX AI by JagX & JRILICENSE.

You are helpful, clear, and excellent at coding and explanations.
"""

# ====================== INVISIBLE WATERMARK ======================
def add_invisible_watermark(text: str) -> str:
    """
    Adds an invisible watermark using zero-width characters.
    This watermark is not visible but usually survives copy-paste.
    """
    ZWSP = "\u200B"   # Zero Width Space
    ZWNJ = "\u200C"   # Zero Width Non-Joiner
    ZWJ  = "\u200D"   # Zero Width Joiner

    # Secret pattern representing "JAGX"
    pattern = [ZWNJ, ZWSP, ZWSP, ZWSP, ZWJ, ZWJ, ZWJ, ZWSP]
    watermark = "".join(pattern)

    if not text:
        return watermark

    if len(text) < 15:
        return text + watermark

    # Insert watermark in multiple places for better survival
    third = len(text) // 3
    text = (
        text[:3] + watermark +
        text[3:third] + watermark +
        text[third:third*2] + watermark +
        text[third*2:]
    )
    return text


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
    """Simple free search using DuckDuckGo HTML"""
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None

        texts = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        if not texts:
            texts = re.findall(r'class="result__snippet">(.*?)</td>', r.text, re.DOTALL)

        clean = []
        for t in texts[:3]:
            t = re.sub(r'<.*?>', '', t).strip()
            if t and len(t) > 40:
                clean.append(t)

        if clean:
            return "Based on available information:\n\n" + "\n\n".join(clean)
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
    window = 3600

    rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < window]

    if len(rate_limit_store[key]) >= limit:
        return False, f"Hourly limit reached ({limit} requests/hour). Please wait or upgrade."

    rate_limit_store[key].append(now)
    remaining = limit - len(rate_limit_store[key])
    return True, f"{remaining} requests remaining this hour"


# ====================== LLM ======================
def call_external_llm(messages: list, max_tokens: int = 1200) -> Optional[str]:
    # Try Hugging Face
    if HF_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.4
            }
            r = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except:
            pass

    # Try NVIDIA
    if NVIDIA_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "meta/llama-3.1-8b-instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.4,
                "stream": False
            }
            r = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except:
            pass

    return None


def generate_response(user_message: str, history: Optional[List[Dict]] = None) -> str:
    # 1. Local Knowledge
    local = search_knowledge(user_message)
    if local:
        return local

    # 2. External LLM (if keys exist)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for m in history[-8:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    external = call_external_llm(messages)
    if external:
        return external

    # 3. Free Internet Search
    search_result = free_web_search(user_message)
    if search_result:
        return search_result

    # 4. Final fallback
    return "I am JagX AI, created by JagX & JRILICENSE. I couldn't find a good answer for that right now. Please try rephrasing your question."


# ====================== MODELS ======================
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 1200
    history: Optional[List[Dict[str, str]]] = None


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
        "status": "JagX AI 4.2 is running",
        "version": "4.2.0",
        "created_by": "JagX & JRILICENSE",
        "features": [
            "hourly_rate_limit",
            "key_management",
            "local_knowledge",
            "free_search",
            "invisible_watermark"
        ]
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


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    allowed, quota_msg = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_msg)

    reply = generate_response(req.message, req.history)

    # Add invisible watermark
    reply = add_invisible_watermark(reply)

    return {
        "response": reply,
        "model": "JagX AI 4.2",
        "quota": quota_msg
    }


# ---------- ADMIN KEY MANAGEMENT ----------
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