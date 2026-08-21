"""
JagX AI 6.0
General Purpose AI + Reasoning/Tool-Use + Vision + Image Gen + Supabase Auth
Created by JagX & JRILICENSE
"""

import os
import json
import secrets
import threading
import time
import re
import hmac
import base64
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from urllib.parse import quote
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ====================== CONFIG ======================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# These model names can go stale as providers update their lineups —
# check console.groq.com/docs/models if calls start failing with a 404.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")

PISTON_URL = "https://emkc.org/api/v2/piston"

KEYS_FILE = "keys.json"
KNOWLEDGE_FILE = "jagx_knowledge.json"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(
    k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip()
)

# ===== Supabase (optional — auth layer on top of the existing key system) =====
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_SERVICE_KEY)

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

app = FastAPI(
    title="JagX AI 6.0",
    description="General Purpose AI with reasoning, tools, vision, image gen and auth by JagX & JRILICENSE",
    version="6.0.0"
)

lock = threading.Lock()
rate_limit_store = defaultdict(list)
auth_attempt_store = defaultdict(list)  # crude per-IP brute force guard on auth routes

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend domain once you have one
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENT_SYSTEM_PROMPT = f"""You are JagX AI 6.0 — a powerful general-purpose AI created by JagX and JRILICENSE.

STRICT IDENTITY RULES:
- Your name is JagX AI.
- You were created by JagX and JRILICENSE.
- Never say you were created by OpenAI, Meta, Alibaba, Google, Anthropic, Groq, or any other company.
- Always introduce yourself as JagX AI by JagX & JRILICENSE when asked.

CAPABILITIES:
- Deep reasoning, multi-step planning, mathematics
- Writing, explaining, and debugging code in many languages
- Real sandboxed code execution to verify answers
- Real-time web search
- Image generation and image understanding
- Current date and time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

TOOLS — you may use these when you genuinely need current information or must execute/verify code.
To use a tool, reply with ONLY a raw JSON object, nothing else, in one of these forms:
{{"tool": "web_search", "input": "search query"}}
{{"tool": "run_code", "input": {{"language": "python", "code": "print('hello')"}}}}

When you are ready to answer the user, reply with ONLY:
{{"final": "your complete, well-formatted answer"}}

Rules:
- Don't use a tool unless you actually need it — if you already know the answer, go straight to {{"final": ...}}.
- Never call more tools than necessary.
- Think step by step before deciding, but only the JSON should appear in your reply — no extra commentary outside the JSON.
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

# ====================== KEY / KNOWLEDGE HELPERS ======================
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
    except Exception:
        return None

# ====================== SANDBOXED CODE EXECUTION (Piston) ======================
_piston_runtimes_cache = {"data": None, "ts": 0}

def _get_piston_runtimes():
    now = time.time()
    if _piston_runtimes_cache["data"] and now - _piston_runtimes_cache["ts"] < 3600:
        return _piston_runtimes_cache["data"]
    try:
        r = requests.get(f"{PISTON_URL}/runtimes", timeout=10)
        if r.status_code == 200:
            _piston_runtimes_cache["data"] = r.json()
            _piston_runtimes_cache["ts"] = now
            return _piston_runtimes_cache["data"]
    except Exception:
        pass
    return _piston_runtimes_cache["data"] or []

def _resolve_piston_version(language: str) -> Optional[Tuple[str, str]]:
    runtimes = _get_piston_runtimes()
    language = language.lower().strip()
    aliases = {"py": "python", "js": "javascript", "node": "javascript", "c++": "cpp"}
    language = aliases.get(language, language)
    for rt in runtimes:
        if rt.get("language") == language or language in (rt.get("aliases") or []):
            return rt.get("language"), rt.get("version")
    return None

def run_code_sandboxed(language: str, code: str, stdin: str = "") -> str:
    if not code or not code.strip():
        return "Error: no code provided."
    if len(code) > MAX_CODE_LEN:
        return f"Error: code too long (max {MAX_CODE_LEN} characters)."
    resolved = _resolve_piston_version(language)
    if not resolved:
        return f"Error: unsupported or unrecognized language '{language}'."
    lang, version = resolved
    try:
        payload = {
            "language": lang,
            "version": version,
            "files": [{"content": code}],
            "stdin": stdin or ""
        }
        r = requests.post(f"{PISTON_URL}/execute", json=payload, timeout=20)
        if r.status_code != 200:
            return f"Error: sandbox execution failed (status {r.status_code})."
        data = r.json()
        run = data.get("run", {})
        compile_ = data.get("compile", {})
        out = ""
        if compile_ and compile_.get("stderr"):
            out += f"[compile stderr]\n{compile_['stderr']}\n"
        out += f"[stdout]\n{run.get('stdout', '')}\n"
        if run.get("stderr"):
            out += f"[stderr]\n{run['stderr']}\n"
        out += f"[exit code: {run.get('code')}]"
        return out[:MAX_TOOL_RESULT_LEN]
    except Exception as e:
        return f"Error: sandbox request failed ({str(e)[:200]})."

# ====================== LLM CASCADE ======================
def _extract_openai_style_content(resp_json: dict) -> Optional[str]:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return None

def call_external_llm(messages: list, max_tokens: int = 1500) -> Optional[str]:
    if GROQ_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.5
            }
            r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                content = _extract_openai_style_content(r.json())
                if content:
                    return content
        except Exception:
            pass
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
                content = _extract_openai_style_content(r.json())
                if content:
                    return content
        except Exception:
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
                content = _extract_openai_style_content(r.json())
                if content:
                    return content
        except Exception:
            pass
    return None

def extract_json_action(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and ("tool" in obj or "final" in obj):
            return obj
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and ("tool" in obj or "final" in obj):
                return obj
        except Exception:
            pass
    return None

def run_agent_loop(user_message: str, history: Optional[List[Dict]], max_tokens: int) -> str:
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if history:
        for m in history[-10:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_STEPS):
        raw = call_external_llm(messages, max_tokens=max_tokens)
        if not raw:
            return "I couldn't reach any AI model right now. Please try again shortly."
        action = extract_json_action(raw)
        if not action:
            return raw  # model answered directly without the tool protocol
        if "final" in action:
            return str(action["final"])
        tool = action.get("tool")
        tool_input = action.get("input")
        if tool == "web_search":
            result = free_web_search(str(tool_input)) or "No results found."
        elif tool == "run_code":
            if isinstance(tool_input, dict):
                lang = tool_input.get("language", "python")
                code = tool_input.get("code", "")
            else:
                lang, code = "python", str(tool_input)
            result = run_code_sandboxed(lang, code)
        else:
            result = f"Unknown tool requested: {tool}"
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"[TOOL RESULT for {tool}]\n{result[:MAX_TOOL_RESULT_LEN]}\n\n"
                        f"Continue. If you now have enough info, reply with {{\"final\": \"...\"}}."
        })

    messages.append({"role": "user", "content": "Give your final answer now, as plain text, no JSON."})
    final = call_external_llm(messages, max_tokens=max_tokens)
    return final or "I wasn't able to finish reasoning about that in time. Please try rephrasing."

def generate_response(user_message: str, history: Optional[List[Dict]] = None, max_tokens: int = 1500) -> str:
    local = search_knowledge(user_message)
    if local:
        return local
    reply = run_agent_loop(user_message, history, max_tokens)
    if reply:
        return reply
    search_result = free_web_search(user_message)
    if search_result:
        return search_result
    return "I'm having trouble generating a response right now. Please try again in a moment."

# ====================== VISION ======================
def call_groq_vision(images_b64: List[str], question: str, max_tokens: int = 1000) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    try:
        content = [{"type": "text", "text": question}]
        for img in images_b64[:MAX_IMAGES]:
            data_url = img if img.startswith("data:") else f"data:image/jpeg;base64,{img}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": GROQ_VISION_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.4
        }
        r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=70)
        if r.status_code == 200:
            return _extract_openai_style_content(r.json())
    except Exception:
        pass
    return None

def analyze_image_with_vision(images: List[str], question: str, max_tokens: int = 1000) -> str:
    cleaned_images = []
    for img in images[:MAX_IMAGES]:
        cleaned = img.split(",", 1)[-1] if img.startswith("data:") else img
        cleaned_images.append(cleaned)

    groq_answer = call_groq_vision(cleaned_images, question, max_tokens)
    if groq_answer:
        if len(cleaned_images) > 1:
            groq_answer += f"\n\n(Analyzed {len(cleaned_images)} images.)"
        return groq_answer

    if HF_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            img_bytes = base64.b64decode(cleaned_images[0])
            payload = {"inputs": {"image": base64.b64encode(img_bytes).decode(), "question": question}}
            r = requests.post(
                "https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base",
                headers=headers,
                json=payload,
                timeout=45
            )
            if r.status_code == 200:
                result = r.json()
                answer = None
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get("answer") or str(result[0])
                elif isinstance(result, dict):
                    answer = result.get("answer") or result.get("generated_text") or str(result)
                if answer:
                    if len(cleaned_images) > 1:
                        answer += f"\n\n(Note: You uploaded {len(cleaned_images)} images. I analyzed the first one.)"
                    return answer
        except Exception as e:
            print("Vision error:", e)

    if len(cleaned_images) > 1:
        return (
            f"I successfully received {len(cleaned_images)} images. "
            "Please describe the images or ask a specific question."
        )
    return (
        "I successfully received your image. "
        "Please describe the image or ask a specific question."
    )

# ====================== SECURITY HELPERS ======================
def safe_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a or "", b or "")

def check_admin_secret(provided: str):
    if not safe_compare(provided, ADMIN_SECRET):
        raise HTTPException(status_code=403, detail="Invalid admin secret")

def client_ip(request: Request) -> str:
    return request.client.host if request and request.client else "unknown"

def guard_auth_abuse(ip: str):
    now = time.time()
    auth_attempt_store[ip] = [t for t in auth_attempt_store[ip] if now - t < 900]  # 15 min window
    if len(auth_attempt_store[ip]) >= 10:
        raise HTTPException(status_code=429, detail="Too many auth attempts. Try again later.")
    auth_attempt_store[ip].append(now)

# ====================== SUPABASE HELPERS ======================
def supabase_signup(email: str, password: str) -> dict:
    r = requests.post(
        f"{SUPABASE_URL}/auth/v1/signup",
        headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
        json={"email": email, "password": password},
        timeout=15
    )
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=400, detail="Signup failed. Email may already be in use or invalid.")
    return r.json()

def supabase_login(email: str, password: str) -> dict:
    r = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
        json={"email": email, "password": password},
        timeout=15
    )
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return r.json()

def supabase_get_user(access_token: str) -> dict:
    r = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {access_token}"},
        timeout=15
    )
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    return r.json()

def supabase_upsert_profile(user_id: str, email: str, api_key: str, tier: str = "free"):
    requests.post(
        f"{SUPABASE_URL}/rest/v1/profiles",
        headers={
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        },
        json={"id": user_id, "email": email, "api_key": api_key, "tier": tier, "active": True},
        timeout=15
    )

def supabase_get_profile_by_user(user_id: str) -> Optional[dict]:
    r = requests.get(
        f"{SUPABASE_URL}/re