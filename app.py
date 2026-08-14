"""
JagX AI 3.9 — Upgraded Key System + Limits
Created by JagX & JRILICENSE
"""

import os
import json
import secrets
import threading
import base64
import time
import glob
from urllib.parse import quote
from typing import List, Optional, Dict, Any, Union, Tuple

import requests
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ====================== CONFIG ======================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

TIER_MODELS = {
    "fast": {
        "hf": ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
        "nvidia": ["qwen/qwen2.5-7b-instruct", "meta/llama-3.2-3b-instruct"]
    },
    "balanced": {
        "hf": ["Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct"],
        "nvidia": ["qwen/qwen2.5-coder-32b-instruct", "meta/llama-3.1-70b-instruct"]
    },
    "expert": {
        "hf": ["Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct"],
        "nvidia": ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "qwen/qwen2.5-coder-32b-instruct"]
    },
    "heavy": {
        "hf": ["Qwen/Qwen2.5-Coder-14B-Instruct"],
        "nvidia": ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "qwen/qwen2.5-coder-32b-instruct"]
    },
    "auto": {
        "hf": ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-14B-Instruct"],
        "nvidia": ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "qwen/qwen2.5-coder-32b-instruct"]
    }
}

DEFAULT_TIER = "auto"

KEYS_FILE = "keys.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip())

# Daily message limits
TIER_LIMITS = {
    "free": 90,
    "premium": 1000,
    "premium_plus": 3000,
    "master": None,   # Unlimited
    "admin": None     # Unlimited
}

app = FastAPI(
    title="JagX AI 3.9",
    description="JagX AI with Upgraded Key System + Limits by JagX & JRILICENSE",
    version="3.9.0"
)
lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are JagX AI 3.9 — an elite independent AI created by JagX & JRILICENSE.

STRICT IDENTITY RULES:
- Your name is JagX AI.
- You were created by JagX & JRILICENSE.
- Never say you were made by Alibaba, Qwen, Meta, OpenAI, Google, NVIDIA or any other company.
- Always present yourself as JagX AI.

You have conversation memory. When a user says things like "more examples", "continue", "explain more", you must continue from the previous topic.

You are excellent at coding, cybersecurity, mathematics, English, school help, and clear reasoning.
"""

# ====================== KNOWLEDGE ======================
def load_all_knowledge() -> List[Dict]:
    knowledge = []
    files = glob.glob("jagx_knowledge*.json")
    
    if not files:
        default = [
            {"question": "who are you", "answer": "I am JagX AI 3.9, created by JagX & JRILICENSE."},
            {"question": "who created you", "answer": "I was created by JagX & JRILICENSE."}
        ]
        with open("jagx_knowledge.json", "w") as f:
            json.dump(default, f, indent=2)
        return default

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    knowledge.extend(data)
        except Exception:
            pass
    return knowledge

def search_knowledge(query: str) -> Optional[str]:
    knowledge = load_all_knowledge()
    query_lower = query.lower().strip()
    if not query_lower:
        return None

    for item in knowledge:
        q = item.get("question", "").lower()
        if q == query_lower or q in query_lower or query_lower in q:
            return item.get("answer")

    query_words = set(query_lower.split())
    best_score = 0
    best_answer = None

    for item in knowledge:
        q = item.get("question", "").lower()
        q_words = set(q.split())
        score = len(query_words.intersection(q_words))
        if score > best_score and score >= 1:
            best_score = score
            best_answer = item.get("answer")
    return best_answer

# ====================== KEYS SYSTEM ======================
def load_keys():
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "w") as f:
            json.dump({}, f)
    with open(KEYS_FILE, "r") as f:
        return json.load(f)

def save_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def get_today() -> str:
    return time.strftime("%Y-%m-%d")

def is_valid_key(key: str) -> bool:
    if not key:
        return False
    if key in PERMANENT_KEYS:
        return True
    keys = load_keys()
    return key in keys and keys[key].get("active", True)

def check_and_consume_quota(key: str) -> Tuple[bool, str]:
    """Returns (allowed, message)"""
    if key in PERMANENT_KEYS:
        return True, "unlimited (master)"

    keys = load_keys()
    if key not in keys:
        return False, "Invalid key"

    user = keys[key]

    if not user.get("active", True):
        return False, "This API key has been blocked"

    tier = user.get("tier", "free")
    limit = TIER_LIMITS.get(tier)

    if limit is None:  # master / admin
        return True, "unlimited"

    today = get_today()
    usage = user.get("usage", {})

    if usage.get("date") != today:
        usage = {"date": today, "count": 0}

    if usage["count"] >= limit:
        return False, f"Daily limit reached ({limit} messages). Please upgrade your plan."

    usage["count"] += 1
    user["usage"] = usage
    keys[key] = user
    save_keys(keys)

    remaining = limit - usage["count"]
    return True, f"{remaining} messages remaining today"

# ====================== MODELS ======================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 2000
    temperature: float = 0.3
    tier: Optional[str] = "auto"
    history: Optional[List[ChatMessage]] = None
    system: Optional[str] = None

class CreateKeyRequest(BaseModel):
    owner_label: str
    admin_secret: str

class UpgradeKeyRequest(BaseModel):
    api_key: str
    new_tier: str
    admin_secret: str

class BlockKeyRequest(BaseModel):
    api_key: str
    active: bool
    admin_secret: str

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False
    tier: Optional[str] = "auto"

class KnowledgeAddRequest(BaseModel):
    question: str
    answer: str
    admin_secret: str

# ====================== LLM ======================
def call_llm(messages: list, max_tokens: int = 2000, temperature: float = 0.3, tier: str = "auto") -> str:
    tier = (tier or "auto").lower()
    if tier not in TIER_MODELS:
        tier = "auto"

    models_config = TIER_MODELS[tier]
    errors = []

    # Try Hugging Face
    if HF_TOKEN:
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        for model in models_config["hf"]:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": min(max_tokens, 4096),
                "temperature": temperature
            }
            try:
                r = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=90)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                errors.append(f"HF/{model}: {r.status_code}")
            except Exception as e:
                errors.append(f"HF/{model}: {str(e)[:50]}")

    # Try NVIDIA
    if NVIDIA_API_KEY:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        for model in models_config["nvidia"]:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": min(max_tokens, 4096),
                "temperature": temperature,
                "stream": False
            }
            try:
                r = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=90)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                errors.append(f"NVIDIA/{model}: {r.status_code}")
            except Exception as e:
                errors.append(f"NVIDIA/{model}: {str(e)[:50]}")

    # Local knowledge fallback
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    local = search_knowledge(last_user)
    if local:
        return local

    return "I'm JagX AI. External providers are temporarily unavailable and I don't have a matching answer in my local knowledge yet. Please try again shortly."

# ====================== ROUTES ======================
@app.get("/")
def root():
    return {
        "status": "JagX AI 3.9 is running",
        "version": "3.9.0",
        "tiers": list(TIER_MODELS.keys()),
        "key_tiers": list(TIER_LIMITS.keys()),
        "knowledge_files": glob.glob("jagx_knowledge*.json"),
        "providers": {
            "huggingface": bool(HF_TOKEN),
            "nvidia": bool(NVIDIA_API_KEY)
        },
        "created_by": "JagX & JRILICENSE"
    }

@app.post("/create-key")
def create_key(req: CreateKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    with lock:
        keys = load_keys()
        new_key = "jagx-" + secrets.token_hex(16)
        keys[new_key] = {
            "owner": req.owner_label,
            "active": True,
            "tier": "free",
            "usage": {"date": get_today(), "count": 0},
            "created": time.time()
        }
        save_keys(keys)

    return {
        "api_key": new_key,
        "owner": req.owner_label,
        "tier": "free",
        "daily_limit": 90
    }

@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    allowed, quota_msg = check_and_consume_quota(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_msg)

    system = req.system or SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]

    if req.history:
        for m in req.history[-12:]:
            if m.role in ("user", "assistant") and m.content:
                messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": req.message})

    reply = call_llm(messages, req.max_tokens, req.temperature, req.tier or DEFAULT_TIER)

    return {
        "response": reply,
        "tier_used": req.tier or DEFAULT_TIER,
        "model": "JagX AI",
        "version": "3.9",
        "quota": quota_msg
    }

@app.post("/v1/chat/completions")
def openai_compatible(
    req: OpenAIChatRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    key = x_api_key
    if not key and authorization:
        key = authorization.replace("Bearer ", "").strip()

    if not is_valid_key(key or ""):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    allowed, quota_msg = check_and_consume_quota(key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_msg)

    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    reply = call_llm(
        messages,
        req.max_tokens or 2000,
        req.temperature if req.temperature is not None else 0.3,
        req.tier or DEFAULT_TIER
    )

    return {
        "id": f"jagx-{secrets.token_hex(8)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "JagX AI 3.9",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "quota": quota_msg
    }

# ====================== ADMIN ROUTES ======================
@app.post("/admin/upgrade-key")
def upgrade_key(req: UpgradeKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    if req.new_tier not in TIER_LIMITS:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Use: {list(TIER_LIMITS.keys())}")

    with lock:
        keys = load_keys()
        if req.api_key not in keys:
            raise HTTPException(status_code=404, detail="API key not found")

        keys[req.api_key]["tier"] = req.new_tier
        keys[req.api_key]["usage"] = {"date": get_today(), "count": 0}
        save_keys(keys)

    return {
        "success": True,
        "api_key": req.api_key,
        "new_tier": req.new_tier,
        "daily_limit": TIER_LIMITS[req.new_tier]
    }

@app.post("/admin/block-key")
def block_key(req: BlockKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    with lock:
        keys = load_keys()
        if req.api_key not in keys:
            raise HTTPException(status_code=404, detail="API key not found")

        keys[req.api_key]["active"] = req.active
        save_keys(keys)

    status = "unblocked" if req.active else "blocked"
    return {"success": True, "api_key": req.api_key, "status": status}

@app.get("/admin/key-info")
def key_info(api_key: str, admin_secret: str):
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    if api_key in PERMANENT_KEYS:
        return {
            "api_key": api_key,
            "tier": "master",
            "active": True,
            "usage": "unlimited",
            "daily_limit": None
        }

    keys = load_keys()
    if api_key not in keys:
        raise HTTPException(status_code=404, detail="API key not found")

    info = keys[api_key]
    tier = info.get("tier", "free")
    return {
        "api_key": api_key,
        "owner": info.get("owner"),
        "tier": tier,
        "active": info.get("active", True),
        "usage": info.get("usage", {}),
        "daily_limit": TIER_LIMITS.get(tier)
    }

@app.post("/knowledge/add")
def add_knowledge(req: KnowledgeAddRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    with lock:
        knowledge = []
        if os.path.exists("jagx_knowledge.json"):
            with open("jagx_knowledge.json", "r", encoding="utf-8") as f:
                knowledge = json.load(f)
        knowledge.append({
            "question": req.question.strip(),
            "answer": req.answer.strip()
        })
        with open("jagx_knowledge.json", "w", encoding="utf-8") as f:
            json.dump(knowledge, f, indent=2)

    return {"success": True}

# ====================== START ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)