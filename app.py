"""
JagX AI 6.6 (Fixed & Complete)
General Purpose AI + Fast/Reasoning Dual-Path Routing + Identity Protection
+ Vision + Image Gen + File/Link Reading + PDF/CV Generation
+ Job Search & Application Drafting + Neon Postgres Persistence
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
import uuid
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
from pypdf import PdfReader
import docx
from fpdf import FPDF

# Optional Neon Postgres
try:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import pool as pg_pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

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
KNOWLEDGE_FILE = "jagx_knowledge.json"
DRAFTS_FILE = "jagx_job_drafts.json"
TRAINING_DATA_FILE = "jagx_training_data.jsonl"

DATABASE_URL = os.environ.get("DATABASE_URL", "")
DB_ENABLED = bool(DATABASE_URL) and PSYCOPG2_AVAILABLE

ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
PERMANENT_KEYS = set(k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip())

TIER_HOURLY_LIMITS = {
    "free": 60,
    "premium": 300,
    "premium_plus": 800,
    "master": None,
    "admin": None
}

MAX_MESSAGE_LEN = 8000
MAX_CODE_LEN = 20000
MAX_TOOL_RESULT_LEN = 4000
MAX_IMAGES = 6
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_LINK_BYTES = 3 * 1024 * 1024
MAX_EXTRACTED_TEXT_LEN = 12000
MAX_PDF_SECTIONS = 50
MAX_BODY_BYTES = 20 * 1024 * 1024
GLOBAL_IP_RPM = int(os.environ.get("JAGX_GLOBAL_IP_RPM", "120"))
DRAFT_EXPIRY_SECONDS = 48 * 3600

APP_START_TIME = time.time()

app = FastAPI(
    title="JagX AI 6.6",
    description="General Purpose AI by JagX & JRILICENSE",
    version="6.6.0"
)

lock = threading.Lock()
drafts_lock = threading.Lock()
rate_limit_store = defaultdict(list)
global_ip_rate_store = defaultdict(list)

# ====================== SECURITY MIDDLEWARE ======================
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
                global_ip_rate_store[ip] = timestamps
                return JSONResponse(status_code=429, content={"detail": "Too many requests. Slow down."})
            timestamps.append(now)
            global_ip_rate_store[ip] = timestamps
        return await call_next(request)

app.add_middleware(GlobalIPRateLimitMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP = _build_session()

# ====================== SYSTEM PROMPTS ======================
AGENT_SYSTEM_PROMPT = f"""You are JagX AI 6.6 — a powerful general-purpose AI created by JagX and JRILICENSE.

STRICT IDENTITY RULES:
- Your name is JagX AI.
- You were created by JagX and JRILICENSE.
- Never say you were created by OpenAI, Meta, Google, Anthropic, Groq, OpenRouter, Hugging Face, NVIDIA or any other company.
- Always introduce yourself as JagX AI by JagX & JRILICENSE when asked.

Current date and time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

Be clear, helpful, natural and professional.
"""

# ====================== IDENTITY PROTECTION ======================
IDENTITY_LEAK_PATTERNS = [
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
    for pattern, replacement in IDENTITY_LEAK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

# ====================== WATERMARK ======================
def add_invisible_watermark(text: str) -> str:
    ZWSP, ZWNJ, ZWJ = "\u200B", "\u200C", "\u200D"
    watermark = "".join([ZWNJ, ZWSP, ZWSP, ZWSP, ZWJ, ZWJ, ZWJ, ZWSP])
    if not text or len(text) < 15:
        return (text or "") + watermark
    third = len(text) // 3
    return text[:3] + watermark + text[3:third] + watermark + text[third:third*2] + watermark + text[third*2:]

# ====================== DATABASE (Optional Neon) ======================
db_pool = None
if DB_ENABLED:
    try:
        db_pool = pg_pool.SimpleConnectionPool(1, 5, DATABASE_URL, sslmode="require")
        logger.info("Connected to Neon Postgres")
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        DB_ENABLED = False

def get_conn():
    return db_pool.getconn()

def put_conn(conn):
    db_pool.putconn(conn)

# ====================== KEY MANAGEMENT ======================
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
    return True, f"{limit - len(rate_limit_store[key])} requests remaining this hour"

# ====================== SEARCH ======================
def free_web_search(query: str) -> Optional[str]:
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        r = HTTP.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if r.status_code != 200:
            return None
        texts = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        clean = [re.sub(r'<.*?>', '', t).strip() for t in texts[:4] if len(re.sub(r'<.*?>', '', t).strip()) > 40]
        return "Here's what I found:\n\n" + "\n\n".join(clean) if clean else None
    except Exception as e:
        logger.warning(f"Search failed: {e}")
        return None

# ====================== LLM CASCADE ======================
def call_external_llm(messages: list, max_tokens: int = 1500) -> Optional[str]:
    # 1. OpenRouter
    if OPENROUTER_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://jagx.ai",
                "X-Title": "JagX AI"
            }
            payload = {"model": OPENROUTER_MODEL, "messages": messages, "max_tokens": max_tokens}
            r = HTTP.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OpenRouter failed: {e}")

    # 2. Groq
    if GROQ_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.5}
            r = HTTP.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    # 3. Hugging Face
    if HF_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {"model": "Qwen/Qwen2.5-7B-Instruct", "messages": messages, "max_tokens": max_tokens}
            r = HTTP.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"HF failed: {e}")

    # 4. NVIDIA
    if NVIDIA_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "meta/llama-3.1-8b-instruct", "messages": messages, "max_tokens": max_tokens, "stream": False}
            r = HTTP.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"NVIDIA failed: {e}")

    return None

def generate_response(user_message: str, history: Optional[List[Dict]] = None) -> str:
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if history:
        for m in history[-8:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    result = call_external_llm(messages)
    if result:
        return sanitize_identity(result)

    search = free_web_search(user_message)
    if search:
        return search

    return "I am JagX AI, created by JagX & JRILICENSE. I couldn't find a complete answer right now."

# ====================== VISION ======================
def analyze_image_with_vision(images: List[str], question: str) -> str:
    if not images:
        return "No images provided."
    main_image = images[0].split(",")[-1] if "," in images[0] else images[0]

    if HF_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            payload = {"image": main_image, "question": question}
            r = HTTP.post("https://api-inference.huggingface.co/models/Salesforce/blip-vqa-base", headers=headers, json=payload, timeout=40)
            if r.status_code == 200:
                result = r.json()
                if isinstance(result, list) and result:
                    return result[0].get("answer", str(result[0]))
        except Exception as e:
            logger.warning(f"Vision error: {e}")

    return "I received your image(s). Full advanced vision is still being improved on JagX AI."

# ====================== PDF / CV ======================
def _sanitize_pdf_text(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")

def generate_cv_pdf(cv: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    name = cv.get("name", "Full Name")
    title = cv.get("title", "")
    contact = cv.get("contact", {}) or {}
    summary = cv.get("summary", "")
    experience = cv.get("experience", []) or []
    education = cv.get("education", []) or []
    skills = cv.get("skills", []) or []

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, _sanitize_pdf_text(name), ln=True)

    if title:
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 8, _sanitize_pdf_text(title), ln=True)
        pdf.set_text_color(0, 0, 0)

    contact_line = " | ".join(v for v in [contact.get("email"), contact.get("phone"), contact.get("location")] if v)
    if contact_line:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _sanitize_pdf_text(contact_line))
    pdf.ln(4)

    def section(title_text):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _sanitize_pdf_text(title_text.upper()), ln=True)

    if summary:
        section("Summary")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize_pdf_text(summary))
        pdf.ln(3)

    if experience:
        section("Experience")
        for job in experience:
            role = job.get("role", "")
            company = job.get("company", "")
            dates = job.get("dates", "")
            bullets = job.get("bullets", []) or []
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, _sanitize_pdf_text(f"{role} — {company}"), ln=True)
            if dates:
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, _sanitize_pdf_text(dates), ln=True)
            pdf.set_font("Helvetica", "", 11)
            for b in bullets:
                pdf.multi_cell(0, 6, _sanitize_pdf_text(f"• {b}"))
            pdf.ln(2)

    if education:
        section("Education")
        for edu in education:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, _sanitize_pdf_text(f"{edu.get('degree', '')} — {edu.get('school', '')}"), ln=True)
            if edu.get("dates"):
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, _sanitize_pdf_text(edu.get("dates")), ln=True)
            pdf.ln(1)

    if skills:
        section("Skills")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize_pdf_text(", ".join(skills)))

    return bytes(pdf.output())

# ====================== JOB DRAFTS ======================
def _load_drafts() -> dict:
    if not os.path.exists(DRAFTS_FILE):
        return {}
    try:
        with open(DRAFTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_drafts(drafts: dict):
    tmp = DRAFTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(drafts, f, indent=2)
    os.replace(tmp, DRAFTS_FILE)

def create_draft_record(draft_id: str, subject: str, body: str, job_title: str, company: str):
    with drafts_lock:
        drafts = _load_drafts()
        # Clean expired drafts
        drafts = {
            k: v for k, v in drafts.items()
            if time.time() - v.get("created_at", 0) < DRAFT_EXPIRY_SECONDS
        }
        drafts[draft_id] = {
            "subject": subject,
            "body": body,
            "job_title": job_title,
            "company": company,
            "created_at": time.time()
        }
        _save_drafts(drafts)

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
        "status": "JagX AI 6.6 is running",
        "version": "6.6.0",
        "created_by": "JagX & JRILICENSE",
        "uptime_seconds": int(time.time() - APP_START_TIME),
        "database": "connected" if DB_ENABLED else "file-based"
    }

@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    allowed, quota = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota)

    reply = generate_response(req.message, req.history)
    reply = add_invisible_watermark(reply)
    return {"response": reply, "model": "JagX AI 6.6", "quota": quota}

@app.post("/image")
def generate_image(req: ImageRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    allowed, quota = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota)

    try:
        url = f"https://image.pollinations.ai/prompt/{quote(req.prompt)}?width={req.width}&height={req.height}&nologo=true"
        r = HTTP.get(url, timeout=60)
        if r.status_code == 200:
            return {
                "success": True,
                "image_base64": base64.b64encode(r.content).decode(),
                "format": "png",
                "quota": quota
            }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    raise HTTPException(status_code=502, detail="Image generation failed")

@app.post("/vision")
def vision(req: VisionRequest, x_api_key: str = Header(...)):
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    allowed, quota = check_rate_limit(x_api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota)
    if not req.images:
        raise HTTPException(status_code=400, detail