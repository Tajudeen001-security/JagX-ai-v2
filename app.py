"""
JagX AI 4.0
Clean Key System + Hourly Rate Limiting + Key Management
Created by JagX & JRILICENSE
"""

import os
import json
import secrets
import threading
import time
from typing import Optional, List, Dict, Any
from collections import defaultdict

import requests
from fastapi import FastAPI, HTTPException, Header, Depends
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

# Hourly Rate Limits (requests per hour)
TIER_HOURLY_LIMITS = {
    "free": 60,          # 60 requests per hour
    "premium": 300,
    "premium_plus": 800,
    "master": None,      # Unlimited
    "admin": None
}

app = FastAPI(
    title="JagX AI 4.0",
    description="Clean API with Hourly Rate Limiting + Key Management by JagX & JRILICENSE",
    version="4.0.0"
)

lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are JagX AI 4.0 — an advanced independent AI created by JagX & JRILICENSE.

STRICT IDENTITY RULES:
- Your name is JagX AI.
- You were created by JagX & JRILICENSE.
- Never say you were made by Alibaba, Qwen, Meta, OpenAI, Google, NVIDIA or any other company.
- Always introduce yourself as JagX AI by JagX & JRILICENSE.

You are excellent at coding, cybersecurity, mathematics, reasoning, and clear explanations.
"""

# In-memory rate limit tracker: {api_key: [(timestamp, ...)]}
rate_limit_store = defaultdict(list)


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
        default = [
            {"question": "who are you", "answer": "I am JagX AI 4.0, created by JagX & JRILICENSE."},
            {"question": "who created you", "answer": "I was created by JagX & JRILICENSE."}
        ]
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump(default, f, indent=2)
        return default
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
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


def is_valid_key(key: str) -> bool:
    if not key:
        return False
    if key in PERMANENT_KEYS:
        return True
    keys = load_keys()
    return key in keys and keys[key].get("active", True)


def check_rate_limit(key: str) -> tuple[bool, str]:
    """Returns (allowed, message)"""
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
    window = 3600  # 1 hour

    # Clean old timestamps
    rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < window]

    if len(rate_limit_store[key]) >= limit:
        return False, f"Hourly rate limit reached ({limit} requests/hour). Please wait or upgrade."

    rate_limit_store[key].append(now)
    remaining = limit - len(rate_limit_store[key])
    return True, f"{remaining} requests remaining this hour"


# ====================== MODELS ======================
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 1500
    temperature: float = 0.4
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


# ====================== LLM ======================
def call_llm(messages: list, max_tokens: int = 1500, temperature: float = 0.4) -> str:
    errors = []

    # Try Hugging Face
    if HF_TOKEN:
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": messages,
            "max_tokens": min(max_tokens, 4096),
            "temperature": temperature
        }
        try:
            r = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            errors.append(f"HF: {r.status_code}")
        except Exception as e:
            errors.append(f"HF: {str(e)[:60]}")

    # Try NVIDIA
    if NVIDIA_API_KEY:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": messages,
            "max_tokens": min(max_tokens, 4096),
            "temperature": temperature,
            "stream": False
        }
        try:
            r = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            errors.append(f"NVIDIA: {r.status_code}")
        except Exception as e:
            errors.append(f"NVIDIA: {str(e)[:60]}")

    # Knowledge fallback
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    local = search_knowledge(last_user)
    if local:
        return local

    return "I'm JagX AI. External AI providers are temporarily unavailable. Please try again shortly."


# ====================== ROUTES ======================
@app.get("/")
def root():
    return {
        "status": "JagX AI 4.0 is running",
        "version": "4.0.0",
        "features": [
            "hourly_rate_limiting",
            "key_management",
            "permanent_keys",
            "tiers"
        ],
        "created_by": "JagX & JRILICENSE"
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if req.history:
        for m in req.history[-10:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": req.message})

    reply = call_llm(messages, req.max_tokens, req.temperature)

    return {
        "response": reply,
        "model": "JagX AI 4.0",
        "quota": quota_msg
    }


# ---------- KEY MANAGEMENT ----------
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

    status = "activated" if req.active else "blocked"
    return {"success": True, "message": f"Key {status} successfully"}


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