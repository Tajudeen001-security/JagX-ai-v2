"""
JagX AI 6.8
Restored Features Edition
- Full Tool-Calling System
- PDF / CV Generation
- Dual-path Fast/Reasoning Routing
- Training Data Logging
- Advanced File/Link Reading
- Job Drafts + Identity Protection
Created by JagX & JRILICENSE
"""

import os
import io
import json
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
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import uvicorn
from bs4 import BeautifulSoup
from fpdf import FPDF

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jagx-ai")

# ====================== CONFIG ======================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

PISTON_URL = "https://emkc.org/api/v2/piston"
KEYS_FILE = "keys.json"
DRAFTS_FILE = "jagx_job_drafts.json"
TRAINING_DATA_FILE = "jagx_training_data.jsonl"
TRAINING_DATA_ENABLED = os.environ.get("JAGX_TRAINING_DATA_ENABLED", "true").lower() == "true"

ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip())

TIER_HOURLY_LIMITS = {"free": 60, "premium": 300, "premium_plus": 800, "master": None, "admin": None}

MAX_TOOL_STEPS = 3
MAX_CODE_LEN = 15000
MAX_TOOL_RESULT_LEN = 3500
MAX_BODY_BYTES = 20 * 1024 * 1024
GLOBAL_IP_RPM = 120
DRAFT_EXPIRY_SECONDS = 48 * 3600
APP_START_TIME = time.time()

app = FastAPI(title="JagX AI 6.8", version="6.8.0", description="Created by JagX & JRILICENSE")

lock = threading.Lock()
drafts_lock = threading.Lock()
training_lock = threading.Lock()
rate_limit_store = defaultdict(list)
global_ip_rate_store = defaultdict(list)

# ====================== MIDDLEWARE ======================
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
                return JSONResponse(status_code=429, content={"detail": "Too many requests."})
            timestamps.append(now)
            global_ip_rate_store[ip] = timestamps
        return await call_next(request)

app.add_middleware(GlobalIPRateLimitMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

def _build_session():
    s = requests.Session()
    retries = Retry(total=1, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP = _build_session()

# ====================== PROMPTS ======================
FAST_SYSTEM = f"""You are JagX AI, created by JagX and JRILICENSE. Talk naturally and helpfully.
Never mention OpenAI, Groq, Meta, Google or any other company. You are only JagX AI by JagX & JRILICENSE.
Current date: {datetime.utcnow().strftime("%Y-%m-%d")}."""

AGENT_SYSTEM = f"""You are JagX AI — created by JagX and JRILICENSE.
Never claim to be made by any other company.

You can use tools when needed by replying with ONLY a JSON object:
{{"tool": "web_search", "input": "query"}}
{{"tool": "run_code", "input": {{"language": "python", "code": "print(1+1)"}}}}
{{"tool": "generate_pdf", "input": {{"title": "Title", "sections": [{{"heading": "H1", "content": "text"}}]}}}}
{{"tool": "generate_cv", "input": {{"name": "Name", "title": "Title", "summary": "...", "experience": [], "education": [], "skills": []}}}}

When ready to answer, reply with ONLY:
{{"final": "your answer here"}}

Current date: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
"""

# ====================== IDENTITY + WATERMARK ======================
IDENTITY_PATTERNS = [
    (re.compile(r'\b(openai|chatgpt|gpt-?\d)\b', re.I), "JagX AI"),
    (re.compile(r'\b(anthropic|claude)\b', re.I), "JagX AI"),
    (re.compile(r'\b(llama|meta ai)\b', re.I), "JagX AI"),
    (re.compile(r'\bgroq\b', re.I), "JagX AI"),
    (re.compile(r'\bopenrouter\b', re.I), "JagX AI"),
    (re.compile(r'\bhugging\s?face\b', re.I), "JagX AI"),
    (re.compile(r'\bnvidia\b', re.I), "JagX AI"),
    (re.compile(r'\bqwen\b', re.I), "JagX AI"),
]

def sanitize_identity(text: str) -> str:
    if not text:
        return text
    for p, r in IDENTITY_PATTERNS:
        text = p.sub(r, text)
    return text

def add_invisible_watermark(text: str) -> str:
    ZWSP, ZWNJ, ZWJ = "\u200B", "\u200C", "\u200D"
    wm = "".join([ZWNJ, ZWSP, ZWSP, ZWSP, ZWJ, ZWJ, ZWJ, ZWSP])
    if not text or len(text) < 15:
        return (text or "") + wm
    third = len(text) // 3
    return text[:3] + wm + text[3:third] + wm + text[third:third*2] + wm + text[third*2:]

# ====================== KEYS ======================
def load_keys() -> dict:
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "w") as f:
            json.dump({}, f)
    with open(KEYS_FILE, "r") as f:
        return json.load(f)

def save_keys(keys: dict):
    tmp = KEYS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(keys, f, indent=2)
    os.replace(tmp, KEYS_FILE)

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
        return False, "API key blocked"
    tier = user.get("tier", "free")
    limit = TIER_HOURLY_LIMITS.get(tier)
    if limit is None:
        return True, "unlimited"
    now = time.time()
    rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < 3600]
    if len(rate_limit_store[key]) >= limit:
        return False, f"Hourly limit reached ({limit}/hour)."
    rate_limit_store[key].append(now)
    return True, f"{limit - len(rate_limit_store[key])} remaining this hour"

# ====================== SEARCH + CODE ======================
def free_web_search(query: str) -> str:
    try:
        r = HTTP.get(f"https://html.duckduckgo.com/html/?q={quote(query)}", headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        texts = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        clean = [re.sub(r'<.*?>', '', t).strip() for t in texts[:3] if len(re.sub(r'<.*?>', '', t).strip()) > 40]
        return "Search results:\n\n" + "\n\n".join(clean) if clean else "No good results found."
    except Exception:
        return "Search failed."

def run_code_sandboxed(language: str, code: str) -> str:
    if len(code) > MAX_CODE_LEN:
        return "Code too long."
    try:
        payload = {"language": language.lower(), "version": "*", "files": [{"content": code}]}
        r = HTTP.post(f"{PISTON_URL}/execute", json=payload, timeout=18)
        if r.status_code != 200:
            return f"Execution failed ({r.status_code})"
        data = r.json()
        run = data.get("run", {})
        out = (run.get("stdout") or "") + (run.get("stderr") or "")
        return out[:MAX_TOOL_RESULT_LEN] or "No output"
    except Exception as e:
        return f"Sandbox error: {str(e)[:150]}"

# ====================== PDF / CV ======================
def _sanitize(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")

def generate_pdf_document(title: str, sections: List[Dict]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _sanitize(title))
    pdf.ln(4)
    for sec in sections[:30]:
        if sec.get("heading"):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, _sanitize(sec["heading"]))
        pdf.set_font("Helvetica", "", 11)
        for line in (sec.get("content") or "").split("\n"):
            pdf.multi_cell(0, 6, _sanitize(line))
        pdf.ln(2)
    return bytes(pdf.output())

def generate_cv_pdf(cv: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _sanitize(cv.get("name", "Name")), ln=True)
    if cv.get("title"):
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, _sanitize(cv["title"]), ln=True)
    pdf.ln(4)
    if cv.get("summary"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "SUMMARY", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize(cv["summary"]))
        pdf.ln(3)
    if cv.get("experience"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "EXPERIENCE", ln=True)
        for job in cv["experience"]:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, _sanitize(f"{job.get('role', '')} — {job.get('company', '')}"), ln=True)
            if job.get("dates"):
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, _sanitize(job["dates"]), ln=True)
            pdf.set_font("Helvetica", "", 11)
            for b in job.get("bullets", []):
                pdf.multi_cell(0, 6, _sanitize(f"• {b}"))
            pdf.ln(2)
    if cv.get("skills"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "SKILLS", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize(", ".join(cv["skills"])))
    return bytes(pdf.output())

# ====================== LLM ======================
def call_llm(messages: list, max_tokens: int = 1200) -> Optional[str]:
    # Groq first
    if GROQ_API_KEY:
        try:
            r = HTTP.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                          json={"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.5},
                          timeout=22)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Groq: {e}")

    if OPENROUTER_API_KEY:
        try:
            r = HTTP.post("https://openrouter.ai/api/v1/chat/completions",
                          headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                          json={"model": OPENROUTER_MODEL, "messages": messages, "max_tokens": max_tokens},
                          timeout=25)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OpenRouter: {e}")

    if HF_TOKEN:
        try:
            r = HTTP.post("https://router.huggingface.co/v1/chat/completions",
                          headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                          json={"model": "Qwen/Qwen2.5-7B-Instruct", "messages": messages, "max_tokens": max_tokens},
                          timeout=25)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"HF: {e}")

    return None

def needs_tools(msg: str) -> bool:
    text = (msg or "").lower()
    keywords = ["code", "python", "javascript", "run this", "execute", "pdf", "cv", "resume", "search for", "look up", "latest", "job"]
    return any(k in text for k in keywords)

def extract_json(text: str) -> Optional[dict]:
    try:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip().rstrip("```")
        return json.loads(text)
    except Exception:
        return None

def run_agent(user_message: str, history: Optional[List] = None) -> Tuple[str, Optional[dict]]:
    messages = [{"role": "system", "content": AGENT_SYSTEM}]
    if history:
        for m in history[-5:]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    attachment = None
    for step in range(MAX_TOOL_STEPS):
        raw = call_llm(messages, max_tokens=900)
        if not raw:
            break
        data = extract_json(raw)
        if not data:
            return sanitize_identity(raw), None

        if "final" in data:
            return sanitize_identity(data["final"]), attachment

        tool = data.get("tool")
        inp = data.get("input")

        if tool == "web_search":
            result = free_web_search(str(inp))
        elif tool == "run_code" and isinstance(inp, dict):
            result = run_code_sandboxed(inp.get("language", "python"), inp.get("code", ""))
        elif tool == "generate_pdf" and isinstance(inp, dict):
            try:
                pdf_bytes = generate_pdf_document(inp.get("title", "Document"), inp.get("sections", []))
                b64 = base64.b64encode(pdf_bytes).decode()
                attachment = {"type": "pdf", "filename": "document.pdf", "content_base64": b64}
                result = "PDF generated successfully."
            except Exception as e:
                result = f"PDF failed: {e}"
        elif tool == "generate_cv" and isinstance(inp, dict):
            try:
                pdf_bytes = generate_cv_pdf(inp)
                b64 = base64.b64encode(pdf_bytes).decode()
                attachment = {"type": "pdf", "filename": "cv.pdf", "content_base64": b64}
                result = "CV generated successfully."
            except Exception as e:
                result = f"CV failed: {e}"
        else:
            result = "Unknown tool."

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Tool result:\n{result}"})

    # Fallback
    fallback = call_llm([{"role": "system", "content": FAST_SYSTEM}, {"role": "user", "content": user_message}])
    return sanitize_identity(fallback or "I couldn't complete the request."), attachment

def generate_response(user_message: str, history: Optional[List] = None) -> Tuple[str, Optional[dict]]:
    if needs_tools(user_message):
        return run_agent(user_message, history)
    # Fast path
    messages = [{"role": "system", "content": FAST_SYSTEM}]
    if history:
        for m in history[-6:]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    result = call_llm(messages)
    return sanitize_identity(result or "I couldn't generate a response."), None

# ====================== TRAINING LOG ======================
def log_training(input_text: str, output_text: str):
    if not TRAINING_DATA_ENABLED:
        return
    try:
        with training_lock:
            with open(TRAINING_DATA_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "input": input_text[:4000],
                    "output": output_text[:4000]
                }) + "\n")
    except Exception:
        pass

# ====================== MODELS ======================
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 1200
    history: Optional[List[Dict[str, str]]] = None

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024

class VisionRequest(BaseModel):
    images: List[str]
    question: str = "Describe this image."

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
        "status": "JagX AI 6.8 is running",
        "version": "6.8.0",
        "created_by": "JagX & JRILICENSE",
        "features": ["tool_calling", "pdf_cv", "vision", "image_gen", "dual_path", "training_log"]
    }

@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    allowed, quota = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota)

    reply, attachment = generate_response(req.message, req.history)
    reply = add_invisible_watermark(reply)
    log_training(req.message, reply)

    result = {"response": reply, "model": "JagX AI 6.8", "quota": quota}
    if attachment:
        result["attachment"] = attachment
    return result

@app.post("/image")
def generate_image(req: ImageRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    allowed, quota = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota)
    try:
        url = f"https://image.pollinations.ai/prompt/{quote(req.prompt)}?width={req.width}&height={req.height}&nologo=true"
        r = HTTP.get(url, timeout=40)
        if r.status_code == 200:
            return {"success": True, "image_base64": base64.b64encode(r.content).decode(), "format": "png", "quota": quota}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    raise HTTPException(status_code=502, detail="Image generation failed")

@app.post("/vision")
def vision(req: VisionRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    allowed, quota = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota)
    if not req.images:
        raise HTTPException(status_code=400, detail="No images provided")

    main_image = req.images[0].split(",")[-1] if "," in req.images[0] else req.images[0]
    answer = "I received the image. Advanced vision is limited right now."
    if HF_TOKEN:
        try:
            r = HTTP.post("https://api-in