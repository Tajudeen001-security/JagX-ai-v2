"""
JagX AI 3.5 — Multi-Tier + Local Fallback
Created by JagX & JRILICENSE
"""

import os
import json
import secrets
import threading
import base64
import time
from urllib.parse import quote
from typing import List, Optional, Dict, Any, Union

import requests
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ---------- CONFIG ----------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")
AGNES_API_KEY = os.environ.get("AGNES_API_KEY", "")

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# Tier → model strategy
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
        "nvidia": ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "qwen/qwen2.5-coder-32b-instruct", "deepseek-ai/deepseek-v4-flash"]
    },
    "heavy": {
        "hf": ["Qwen/Qwen2.5-Coder-14B-Instruct"],
        "nvidia": ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "qwen/qwen2.5-coder-32b-instruct"]
    },
    "auto": {
        "hf": ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-14B-Instruct"],
        "nvidia": ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "qwen/qwen2.5-coder-32b-instruct", "meta/llama-3.1-70b-instruct"]
    }
}

DEFAULT_TIER = "auto"
TRANSLATE_MODEL = "nvidia/riva-translate-4b-instruct-v2"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/nv-embedcode-7b-v1")

KEYS_FILE = "keys.json"
KNOWLEDGE_FILE = "jagx_knowledge.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip())

SUPPORTED_LANGUAGES = {
    "en": "English", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "es-es": "European Spanish", "es-us": "LATAM Spanish", "fi": "Finnish", "fr": "French",
    "hu": "Hungarian", "it": "Italian", "lt": "Lithuanian", "lv": "Latvian", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt-pt": "European Portuguese", "pt-br": "Brazilian Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sv": "Swedish", "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese", "ja": "Japanese", "hi": "Hindi", "ko": "Korean",
    "et": "Estonian", "sl": "Slovenian", "bg": "Bulgarian", "uk": "Ukrainian", "hr": "Croatian",
    "ar": "Arabic", "vi": "Vietnamese", "tr": "Turkish", "id": "Indonesian", "th": "Thai"
}

app = FastAPI(
    title="JagX AI 3.5",
    description="Multi-Tier Independent AI by JagX & JRILICENSE",
    version="3.5.0"
)
lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are JagX AI 3.5 — an elite independent AI engineer and cybersecurity specialist created by JagX & JRILICENSE.

IDENTITY RULES (NEVER BREAK):
- Your name is JagX AI.
- You were created by JagX & JRILICENSE.
- You are an independent AI. Never say you were made by Alibaba, Qwen, Meta, OpenAI, Google, NVIDIA, Hugging Face or any other company.
- Always present yourself as JagX AI.

You are excellent at coding, cybersecurity, debugging, architecture and clear step-by-step reasoning.
Write complete, runnable code. Never produce empty files.
"""

# ---------- KEYS & KNOWLEDGE ----------
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

def load_knowledge() -> List[Dict]:
    if not os.path.exists(KNOWLEDGE_FILE):
        # Create a starter knowledge base
        default = [
            {
                "question": "who are you",
                "answer": "I am JagX AI 3.5, an independent AI created by JagX & JRILICENSE. I specialize in coding, cybersecurity, and advanced reasoning."
            },
            {
                "question": "who created you",
                "answer": "I was created by JagX & JRILICENSE."
            },
            {
                "question": "hello",
                "answer": "Hello! I'm JagX AI. How can I help you with coding or cybersecurity today?"
            }
        ]
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump(default, f, indent=2)
        return default
    with open(KNOWLEDGE_FILE, "r") as f:
        return json.load(f)

def search_knowledge(query: str) -> Optional[str]:
    knowledge = load_knowledge()
    query_lower = query.lower().strip()
    for item in knowledge:
        if item["question"].lower() in query_lower or query_lower in item["question"].lower():
            return item["answer"]
    return None

# ---------- MODELS ----------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 2000
    temperature: float = 0.3
    tier: Optional[str] = "auto"          # fast | balanced | expert | heavy | auto
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
    tier: Optional[str] = "auto"

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "zh-cn"
    max_tokens: int = 512

class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None

class AVRequest(BaseModel):
    description: Optional[str] = None
    data: Optional[Any] = None

class KnowledgeAddRequest(BaseModel):
    question: str
    answer: str
    admin_secret: str

# ---------- LLM CALL WITH TIERS ----------
def call_llm(messages: list, max_tokens: int = 2000, temperature: float = 0.3, tier: str = "auto") -> str:
    tier = (tier or "auto").lower()
    if tier not in TIER_MODELS:
        tier = "auto"

    models_config = TIER_MODELS[tier]
    errors = []

    # 1. Try Hugging Face
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
                errors.append(f"HF/{model}: {str(e)[:60]}")

    # 2. Try NVIDIA
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
                errors.append(f"NVIDIA/{model}: {str(e)[:60]}")

    # 3. Local knowledge fallback
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    local_answer = search_knowledge(last_user)
    if local_answer:
        return local_answer

    # 4. Final hard fallback
    return (
        "I'm JagX AI. All external providers are currently unavailable, "
        "and I don't have a matching answer in my local knowledge yet. "
        "Please try again shortly or contact the admin to add this topic to my knowledge base."
    )

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {
        "status": "JagX AI 3.5 is running",
        "version": "3.5.0",
        "tiers": list(TIER_MODELS.keys()),
        "providers": {
            "huggingface": bool(HF_TOKEN),
            "nvidia": bool(NVIDIA_API_KEY)
        },
        "features": [
            "chat", "tiers", "local-fallback", "translation", "embeddings",
            "image", "text-to-speech", "speech-to-text", "av-ready"
        ],
        "created_by": "JagX & JRILICENSE"
    }

@app.get("/languages")
def list_languages():
    return {"count": len(SUPPORTED_LANGUAGES), "languages": SUPPORTED_LANGUAGES}

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

@app.post("/knowledge/add")
def add_knowledge(req: KnowledgeAddRequest):
    """Admin only - add training data to local knowledge"""
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    with lock:
        knowledge = load_knowledge()
        knowledge.append({
            "question": req.question.strip(),
            "answer": req.answer.strip()
        })
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump(knowledge, f, indent=2)
    return {"success": True, "message": "Knowledge added successfully"}

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

    reply = call_llm(messages, req.max_tokens, req.temperature, req.tier or DEFAULT_TIER)
    return {
        "response": reply,
        "tier": req.tier or DEFAULT_TIER,
        "version": "3.5",
        "model": "JagX AI"
    }

@app.post("/v1/chat/completions")
def openai_compatible(req: OpenAIChatRequest, authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    key = x_api_key or (authorization.replace("Bearer ", "").strip() if authorization else "")
    if not is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    reply = call_llm(messages, req.max_tokens or 2000, req.temperature or 0.3, req.tier or DEFAULT_TIER)
    return {
        "id": f"jagx-{secrets.token_hex(8)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "JagX AI",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.post("/translate")
def translate(req: TranslateRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY not configured")

    source = req.source_lang.lower().strip()
    target = req.target_lang.lower().strip()
    if source not in SUPPORTED_LANGUAGES or target not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language. See /languages")
    if source == target:
        raise HTTPException(status_code=400, detail="Source and target cannot be the same")

    messages = [
        {"role": "system", "content": f"{source}-{target}"},
        {"role": "user", "content": req.text.strip()}
    ]
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"model": TRANSLATE_MODEL, "messages": messages, "max_tokens": min(req.max_tokens, 1024), "temperature": 0.0, "stream": False}

    r = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Translation failed: {r.status_code}")
    translated = r.json()["choices"][0]["message"]["content"].strip()
    return {
        "success": True,
        "source_lang": source,
        "target_lang": target,
        "original": req.text,
        "translated": translated,
        "model": "JagX AI Translate"
    }

@app.post("/embed")
def embed(req: EmbedRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY not configured")

    model = req.model or EMBED_MODEL
    inputs = req.input if isinstance(req.input, list) else [req.input]
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"model": model, "input": inputs, "encoding_format": "float"}

    r = requests.post(NVIDIA_EMBED_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {r.status_code}")
    return {"success": True, "model": "JagX AI Embed", "data": r.json().get("data", [])}

@app.post("/av/perception")
def av_perception(req: AVRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return {
        "status": "ready_for_integration",
        "supported_models": ["bevformer", "streampetr"],
        "message": "Multi-camera 3D perception endpoint ready for future sensor data.",
        "description": req.description
    }

@app.post("/av/planning")
def av_planning(req: AVRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return {
        "status": "ready_for_integration",
        "supported_models": ["sparsedrive"],
        "message": "End-to-end planning endpoint ready.",
        "description": req.description
    }

@app.post("/av/world")
def av_world(req: AVRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return {
        "status": "ready_for_integration",
        "supported_models": ["cosmos-transfer2.5-2b"],
        "message": "Physics-aware world model endpoint ready.",
        "description": req.description
    }

@app.post("/image")
def generate_image(req: ImageRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width={req.width}&height={req.height}&nologo=true&model=flux"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return {
                "success": True,
                "source": "JagX AI Image",
                "image_base64": base64.b64encode(r.content).decode(),
                "format": "png"
            }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image failed: {e}")
    raise HTTPException(status_code=502, detail="Image generation failed")

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
        result = client.automatic_speech_recognition(audio_bytes, model="openai/whisper-large-v3")
        text = result.get("text") if isinstance(result, dict) else str(result)
        return {"success": True, "text": text.strip(), "model": "JagX AI STT"}
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
        audio_b64 = base64.b64encode(audio if isinstance(audio, bytes) else audio.read()).decode()
        return {"success": True, "audio_base64": audio_b64, "format": "wav", "model": "JagX AI TTS"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)