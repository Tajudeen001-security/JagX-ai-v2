"""
JagX Backend v1.1.3 — fast text+coding API
Created by JagX & JRILICENSE
"""
import os, json, secrets, threading, time, re, logging
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from urllib.parse import quote
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jagx-backend")

GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
HF_TOKEN = (os.environ.get("HF_TOKEN") or "").strip().strip('"').strip("'")
OPENROUTER_API_KEY = (os.environ.get("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")

GROQ_MODEL = (os.environ.get("GROQ_MODEL") or "llama-3.1-8b-instant").split(",")[0].strip()
OPENROUTER_MODEL = (os.environ.get("OPENROUTER_MODEL") or "meta-llama/llama-3.1-8b-instruct:free").split(",")[0].strip()
HF_MODEL = (os.environ.get("HF_MODEL") or "Qwen/Qwen2.5-7B-Instruct").split(",")[0].strip()

LLM_TIMEOUT = float(os.environ.get("JAGX_LLM_TIMEOUT", "12"))
MAX_OUTPUT_TOKENS = min(1500, int(os.environ.get("JAGX_MAX_OUTPUT_TOKENS", "800") or "800"))
TEMPERATURE = float(os.environ.get("JAGX_TEMPERATURE", "0.3"))

PISTON_URL = "https://emkc.org/api/v2/piston"
KEYS_FILE = "keys.json"
TRAINING_DATA_FILE = "jagx_training_data.jsonl"
TRAINING_DATA_ENABLED = os.environ.get("JAGX_TRAINING_DATA_ENABLED", "true").lower() == "true"
ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret-now")
PERMANENT_KEYS = set(k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip())
TIER_HOURLY_LIMITS = {"free": 80, "premium": 350, "premium_plus": 900, "master": None, "admin": None}
MAX_CODE_LEN = 12000
MAX_TOOL_RESULT_LEN = 3000
MAX_BODY_BYTES = 8 * 1024 * 1024
GLOBAL_IP_RPM = 150
APP_START = time.time()
_LAST_LLM_ERRORS: list = []

app = FastAPI(title="JagX Backend v1.1.3", version="1.1.3", description="Built by JagX & JRILICENSE")
lock = threading.Lock()
training_lock = threading.Lock()
rate_limit_store: Dict[str, List[float]] = defaultdict(list)
global_ip_rate_store: Dict[str, List[float]] = defaultdict(list)

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request body too large."})
        return await call_next(request)

class GlobalIPRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        with lock:
            timestamps = [t for t in global_ip_rate_store[ip] if now - t < 60]
            if len(timestamps) >= GLOBAL_IP_RPM:
                return JSONResponse(status_code=429, content={"detail": "Too many requests from this IP."})
            timestamps.append(now)
            global_ip_rate_store[ip] = timestamps
        return await call_next(request)

app.add_middleware(GlobalIPRateLimitMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=800)

def _build_session():
    s = requests.Session()
    retries = Retry(total=1, backoff_factor=0.25, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP = _build_session()

SYSTEM_PROMPT = f"""You are JagX AI, created by JagX and JRILICENSE.
Be accurate and direct. For coding: working code first. Never invent facts.
Never mention OpenAI, ChatGPT, Groq, Meta, Google, Anthropic or Hugging Face.
You are only JagX AI by JagX & JRILICENSE.
Date: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}."""

IDENTITY_PATTERNS = [
    (re.compile(r"\b(openai|chatgpt|gpt-?\d)\b", re.I), "JagX AI"),
    (re.compile(r"\b(anthropic|claude)\b", re.I), "JagX AI"),
    (re.compile(r"\b(llama|meta ai)\b", re.I), "JagX AI"),
    (re.compile(r"\bgroq\b", re.I), "JagX AI"),
    (re.compile(r"\bopenrouter\b", re.I), "JagX AI"),
    (re.compile(r"\bhugging\s?face\b", re.I), "JagX AI"),
    (re.compile(r"\bnvidia\b", re.I), "JagX AI"),
    (re.compile(r"\bqwen\b", re.I), "JagX AI"),
]

def sanitize_identity(text: str) -> str:
    if not text: return text
    for p, r in IDENTITY_PATTERNS: text = p.sub(r, text)
    return text

def add_invisible_watermark(text: str) -> str:
    ZWSP, ZWNJ, ZWJ = "\u200B", "\u200C", "\u200D"
    wm = "".join([ZWNJ, ZWSP, ZWSP, ZWSP, ZWJ, ZWJ, ZWJ, ZWSP])
    if not text or len(text) < 15: return (text or "") + wm
    third = len(text) // 3
    return text[:3] + wm + text[3:third] + wm + text[third:third*2] + wm + text[third*2:]

def load_keys() -> dict:
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "w") as f: json.dump({}, f)
    try:
        with open(KEYS_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_keys(keys: dict):
    tmp = KEYS_FILE + ".tmp"
    with open(tmp, "w") as f: json.dump(keys, f, indent=2)
    os.replace(tmp, KEYS_FILE)

def is_valid_key(key: str) -> bool:
    if not key: return False
    if key in PERMANENT_KEYS: return True
    keys = load_keys()
    return key in keys and keys[key].get("active", True)

def check_rate_limit(key: str) -> Tuple[bool, str]:
    if key in PERMANENT_KEYS: return True, "unlimited (permanent key)"
    keys = load_keys()
    if key not in keys: return False, "Invalid API key"
    user = keys[key]
    if not user.get("active", True): return False, "API key is blocked"
    tier = user.get("tier", "free")
    limit = TIER_HOURLY_LIMITS.get(tier)
    if limit is None: return True, "unlimited"
    now = time.time()
    with lock:
        rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < 3600]
        if len(rate_limit_store[key]) >= limit:
            return False, f"Hourly limit reached ({limit}/hour)."
        rate_limit_store[key].append(now)
        remaining = limit - len(rate_limit_store[key])
    return True, f"{remaining} remaining this hour"

def free_web_search(query: str, max_results: int = 5) -> str:
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        r = HTTP.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.status_code != 200: return "Search currently unavailable."
        results = []
        blocks = re.findall(r'class="result__a"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td)>', r.text, re.DOTALL)
        for title, snippet in blocks[:max_results]:
            title = re.sub(r"<.*?>", "", title).strip()
            snippet = re.sub(r"<.*?>", "", snippet).strip()
            if title and len(snippet) > 25: results.append(f"**{title}**\n{snippet}")
        return ("Search results:\n\n" + "\n\n".join(results)) if results else "No relevant results found."
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return "Search failed at the moment."

def run_code_sandboxed(language: str, code: str) -> str:
    if len(code) > MAX_CODE_LEN: return "Code is too long."
    try:
        r = HTTP.post(f"{PISTON_URL}/execute", json={"language": language.lower(), "version": "*", "files": [{"content": code}]}, timeout=12)
        if r.status_code != 200: return f"Execution failed (status {r.status_code})"
        run = r.json().get("run", {})
        out = (run.get("stdout") or "") + (run.get("stderr") or "")
        return (out[:MAX_TOOL_RESULT_LEN] or "No output").strip()
    except Exception as e:
        return f"Sandbox error: {str(e)[:120]}"

def call_llm(messages: list, max_tokens: int = None) -> Optional[str]:
    global _LAST_LLM_ERRORS
    _LAST_LLM_ERRORS = []
    max_tokens = max(1, min(int(max_tokens or MAX_OUTPUT_TOKENS), MAX_OUTPUT_TOKENS, 1500))
    if GROQ_API_KEY:
        for model in [GROQ_MODEL, "llama-3.1-8b-instant"]:
            if not model: continue
            try:
                r = HTTP.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": TEMPERATURE},
                    timeout=LLM_TIMEOUT)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                err = f"Groq {model}: HTTP {r.status_code} {r.text[:160]}"
                _LAST_LLM_ERRORS.append(err); logger.warning(err)
                if r.status_code in (401, 403): break
            except Exception as e:
                err = f"Groq {model}: {type(e).__name__}: {e}"
                _LAST_LLM_ERRORS.append(err); logger.warning(err)
    if OPENROUTER_API_KEY:
        for model in [OPENROUTER_MODEL, "meta-llama/llama-3.1-8b-instruct:free"]:
            if not model: continue
            try:
                r = HTTP.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
                             "HTTP-Referer": "https://jagx-ai-v2.onrender.com", "X-Title": "JagX AI"},
                    json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": TEMPERATURE},
                    timeout=LLM_TIMEOUT + 3)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                err = f"OpenRouter {model}: HTTP {r.status_code} {r.text[:160]}"
                _LAST_LLM_ERRORS.append(err); logger.warning(err)
                if r.status_code in (401, 403): break
            except Exception as e:
                err = f"OpenRouter {model}: {type(e).__name__}: {e}"
                _LAST_LLM_ERRORS.append(err); logger.warning(err)
    if HF_TOKEN:
        try:
            r = HTTP.post("https://router.huggingface.co/v1/chat/completions",
                headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                json={"model": HF_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": TEMPERATURE},
                timeout=LLM_TIMEOUT + 5)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            err = f"HF: HTTP {r.status_code} {r.text[:160]}"
            _LAST_LLM_ERRORS.append(err); logger.warning(err)
        except Exception as e:
            err = f"HF: {type(e).__name__}: {e}"
            _LAST_LLM_ERRORS.append(err); logger.warning(err)
    return None

def load_local_knowledge():
    items = []
    base = os.path.dirname(os.path.abspath(__file__)) or "."
    try: names = sorted(os.listdir(base))
    except Exception: names = []
    for name in names:
        if name.startswith("knowledge_pack") and name.endswith(".json"):
            try:
                with open(os.path.join(base, name), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for row in data:
                        k = row.get("k") or row.get("keywords") or []
                        a = row.get("a") or row.get("answer") or ""
                        if k and a: items.append((list(k), str(a)))
            except Exception as e:
                logger.warning(f"Knowledge pack load failed {name}: {e}")
    if not items:
        items = [(["who are you"], "I am JagX AI, created by JagX and JRILICENSE."), (["2+2"], "4")]
    logger.info(f"Loaded {len(items)} local knowledge entries")
    return items

LOCAL_KNOWLEDGE = load_local_knowledge()

def local_knowledge_answer(msg: str) -> Optional[str]:
    t = (msg or "").lower().strip()
    if not t: return None
    for keywords, answer in LOCAL_KNOWLEDGE:
        if any(k in t for k in keywords): return answer
    m = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*\??\s*", t)
    if m: return str(int(m.group(1)) + int(m.group(2)))
    return None

def generate_response(user_message: str, history: Optional[List] = None) -> str:
    local = local_knowledge_answer(user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for m in history[-6:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                content = str(m.get("content", ""))[:3000]
                if content: messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": user_message[:8000]})
    result = call_llm(messages)
    if result: return sanitize_identity(result)
    if local: return sanitize_identity(local)
    if not (GROQ_API_KEY or OPENROUTER_API_KEY or HF_TOKEN):
        return "No LLM API key configured. Set GROQ_API_KEY in Render."
    details = " | ".join(_LAST_LLM_ERRORS[:3]) if _LAST_LLM_ERRORS else "no details"
    return f"LLM providers failed. Details: {details}"

def log_training(input_text: str, output_text: str):
    if not TRAINING_DATA_ENABLED: return
    try:
        with training_lock:
            with open(TRAINING_DATA_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"input": input_text[:3500], "output": output_text[:3500],
                                    "ts": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Training log failed: {e}")

class CreateKeyRequest(BaseModel):
    owner_label: str = Field(..., min_length=1, max_length=80)
    admin_secret: str
    tier: str = Field(default="free", pattern="^(free|premium|premium_plus|master|admin)$")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12000)
    history: Optional[List[Dict[str, str]]] = None
    run_code: Optional[Dict[str, str]] = None
    search: Optional[str] = None

@app.get("/")
def root():
    return {"status": "JagX Backend v1.1.3 is running", "version": "1.1.3", "created_by": "JagX & JRILICENSE",
            "features": ["permanent_api_keys", "text_and_coding", "local_knowledge_fallback", "llm_debug"]}

@app.get("/health")
def health():
    return {"status": "ok", "service": "jagx-backend-v1", "version": "1.1.3", "uptime_seconds": int(time.time() - APP_START)}

@app.get("/debug/status")
def debug_status():
    return {"groq_key_set": bool(GROQ_API_KEY), "groq_key_len": len(GROQ_API_KEY),
            "groq_key_prefix": (GROQ_API_KEY[:7] + "...") if len(GROQ_API_KEY) > 7 else "missing",
            "openrouter_key_set": bool(OPENROUTER_API_KEY), "hf_token_set": bool(HF_TOKEN),
            "groq_model": GROQ_MODEL, "timeout": LLM_TIMEOUT, "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE, "version": "1.1.3", "local_knowledge_entries": len(LOCAL_KNOWLEDGE)}

@app.get("/debug/llm")
def debug_llm():
    results = {}
    test_messages = [{"role": "user", "content": "Reply with only: OK"}]
    if GROQ_API_KEY:
        try:
            r = HTTP.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": test_messages, "max_tokens": 8, "temperature": 0}, timeout=12)
            results["groq"] = {"status": r.status_code, "body": r.text[:300], "ok": r.status_code == 200}
        except Exception as e:
            results["groq"] = {"status": 0, "body": str(e)[:300], "ok": False}
    else:
        results["groq"] = {"ok": False, "body": "GROQ_API_KEY not set"}
    if OPENROUTER_API_KEY:
        try:
            r = HTTP.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
                         "HTTP-Referer": "https://jagx-ai-v2.onrender.com", "X-Title": "JagX AI"},
                json={"model": OPENROUTER_MODEL, "messages": test_messages, "max_tokens": 8, "temperature": 0}, timeout=15)
            results["openrouter"] = {"status": r.status_code, "body": r.text[:300], "ok": r.status_code == 200}
        except Exception as e:
            results["openrouter"] = {"status": 0, "body": str(e)[:300], "ok": False}
    else:
        results["openrouter"] = {"ok": False, "body": "OPENROUTER_API_KEY not set"}
    return results

@app.post("/create-key")
def create_key(body: CreateKeyRequest):
    if body.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    new_key = "jagx-" + secrets.token_hex(16)
    keys = load_keys()
    keys[new_key] = {"owner": body.owner_label.strip(), "tier": body.tier, "active": True,
                     "created_at": datetime.now(timezone.utc).isoformat(), "never_expires": True}
    save_keys(keys)
    limit = TIER_HOURLY_LIMITS.get(body.tier)
    return {"api_key": new_key, "owner": body.owner_label.strip(), "tier": body.tier,
            "hourly_limit": limit if limit is not None else "unlimited", "never_expires": True,
            "message": "Key is permanent. It will not expire or die."}

@app.post("/chat")
def chat(body: ChatRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    if not is_valid_key(x_api_key or ""):
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Send header: x-api-key")
    ok, msg = check_rate_limit(x_api_key)
    if not ok: raise HTTPException(status_code=429, detail=msg)
    if body.run_code and isinstance(body.run_code, dict):
        result = run_code_sandboxed(body.run_code.get("language", "python"), body.run_code.get("code", ""))
        final = add_invisible_watermark(sanitize_identity(f"Code execution result:\n```\n{result}\n```"))
        log_training(body.message or "run", final)
        return {"response": final, "rate_limit": msg}
    if body.search:
        final = add_invisible_watermark(sanitize_identity(free_web_search(body.search)))
        log_training(body.search, final)
        return {"response": final, "rate_limit": msg}
    reply = add_invisible_watermark(generate_response(body.message, body.history))
    log_training(body.message, reply)
    return {"response": reply, "rate_limit": msg}

@app.get("/keys/me")
def my_key_info(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    if not is_valid_key(x_api_key or ""):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if x_api_key in PERMANENT_KEYS:
        return {"tier": "permanent", "hourly_limit": "unlimited", "active": True, "never_expires": True}
    keys = load_keys()
    info = keys.get(x_api_key, {})
    return {"owner": info.get("owner"), "tier": info.get("tier"), "active": info.get("active", True),
            "never_expires": True, "created_at": info.get("created_at")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")), log_level="info")
