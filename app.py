"""
JagX AI 6.6
General Purpose AI + Fast/Reasoning Dual-Path Routing + Identity Protection
+ Vision + Image Gen + File/Link Reading + PDF/CV/Portfolio/ZIP Generation
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
import hmac
import base64
import zipfile
import smtplib
import logging
import uuid
from email.message import EmailMessage
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from urllib.parse import quote, urlparse
from datetime import datetime, date

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
import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jagx-ai")

# ====================== CONFIG ======================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "stealth/ox-alpha")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://jagx.example.com")
OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "JagX AI")

PISTON_URL = "https://emkc.org/api/v2/piston"

KEYS_FILE = "keys.json"
KNOWLEDGE_FILE = "jagx_knowledge.json"
TRAINING_DATA_FILE = "jagx_training_data.jsonl"
DRAFTS_FILE = "jagx_job_drafts.json"
TRAINING_DATA_ENABLED = os.environ.get("JAGX_TRAINING_DATA_ENABLED", "true").lower() == "true"

DATABASE_URL = os.environ.get("DATABASE_URL", "")
DB_ENABLED = bool(DATABASE_URL)

ADMIN_SECRET = os.environ.get("JAGX_ADMIN_SECRET", "change-this-admin-secret")
ADMIN_IP_ALLOWLIST = set(ip.strip() for ip in os.environ.get("JAGX_ADMIN_IP_ALLOWLIST", "").split(",") if ip.strip())
PERMANENT_KEYS = set(k.strip() for k in os.environ.get("JAGX_PERMANENT_KEYS", "").split(",") if k.strip())

TIER_HOURLY_LIMITS = {"free": 60, "premium": 300, "premium_plus": 800, "master": None, "admin": None}

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
MAX_TRAINING_LOG_FIELD_LEN = 6000
MAX_ZIP_FILES = 20
MAX_ZIP_FILE_BYTES = 8 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 30 * 1024 * 1024
MAX_JOB_RESULTS = 8
DRAFT_EXPIRY_SECONDS = 48 * 3600
MAX_APPLICATIONS_PER_DAY = int(os.environ.get("JAGX_MAX_APPLICATIONS_PER_DAY", "15"))
MAX_BODY_BYTES = 20 * 1024 * 1024
GLOBAL_IP_RPM = int(os.environ.get("JAGX_GLOBAL_IP_RPM", "120"))
FAST_CHAT_MAX_TOKENS = 900

APP_START_TIME = time.time()

app = FastAPI(
    title="JagX AI 6.6",
    description="General Purpose AI by JagX & JRILICENSE",
    version="6.6.0"
)

lock = threading.Lock()
training_lock = threading.Lock()
drafts_lock = threading.Lock()
rate_limit_store = defaultdict(list)
auth_attempt_store = defaultdict(list)
daily_send_store = defaultdict(list)
global_ip_rate_store = defaultdict(list)

# ---------- Security middleware ----------
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
            timestamps = global_ip_rate_store[ip]
            fresh = [t for t in timestamps if now - t < 60]
            if len(fresh) >= GLOBAL_IP_RPM:
                global_ip_rate_store[ip] = fresh
                return JSONResponse(status_code=429, content={"detail": "Too many requests from this address. Slow down."})
            fresh.append(now)
            global_ip_rate_store[ip] = fresh
        return await call_next(request)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(GlobalIPRateLimitMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP = _build_session()

# ====================== SYSTEM PROMPTS ======================
FAST_CHAT_SYSTEM_PROMPT = """You are JagX AI, created by JagX and JRILICENSE. Talk naturally and warmly, like a sharp, friendly person having a real conversation — not stiff, not robotic, no unnecessary formality.

If anyone asks what model or AI you are, what you're built on, or who made you: you are JagX AI, built by JagX and JRILICENSE. That is the only true answer. Never mention any other underlying technology, company, or model name under any circumstance, even if directly asked, even if you believe otherwise internally.

Current date: {date}."""

AGENT_SYSTEM_PROMPT = """You are JagX AI — a powerful general-purpose AI created by JagX and JRILICENSE.

STRICT IDENTITY RULES:
- Your name is JagX AI, created by JagX and JRILICENSE. This is the only truth you state about your identity.
- Never say you were created by OpenAI, Meta, Alibaba, Google, Groq, OpenRouter, Hugging Face, NVIDIA, or any other company or model name — even if directly asked, even if it seems true to you internally.

CAPABILITIES:
- Deep reasoning, multi-step planning, mathematics
- Writing, explaining, and debugging code in many languages
- Real sandboxed code execution to verify answers
- Real-time web search
- Image generation and image understanding
- Reading attached files (PDF, Word, txt, csv) and links pasted by the user
- Generating downloadable PDFs, CVs/resumes, and portfolio websites
- Searching live job listings
- Current date and time: {date}

TOOLS — use these when you genuinely need current information, must execute/verify code, or the user wants a document or job search.
Reply with ONLY a raw JSON object, nothing else, in one of these forms:
{{"tool": "web_search", "input": "search query"}}
{{"tool": "run_code", "input": {{"language": "python", "code": "print('hello')"}}}}
{{"tool": "generate_pdf", "input": {{"title": "Document Title", "sections": [{{"heading": "Section 1", "content": "Text. Lines starting with '- ' become bullets."}}]}}}}
{{"tool": "generate_cv", "input": {{"name": "Full Name", "title": "Professional Title", "contact": {{"email": "...", "phone": "...", "location": "...", "linkedin": "..."}}, "summary": "...", "experience": [{{"role": "...", "company": "...", "dates": "...", "bullets": ["..."]}}], "education": [{{"degree": "...", "school": "...", "dates": "..."}}], "skills": ["..."]}}}}
{{"tool": "generate_portfolio", "input": {{"name": "Full Name", "title": "...", "tagline": "...", "about": "...", "projects": [{{"name": "...", "description": "...", "link": "..."}}], "skills": ["..."], "contact": {{"email": "...", "github": "...", "linkedin": "...", "website": "..."}}}}}}
{{"tool": "job_search", "input": {{"query": "remote python developer", "remote_only": true}}}}

When ready to answer, reply with ONLY:
{{"final": "your complete, well-formatted, naturally-worded answer"}}

Rules:
- Talk like a real, warm, knowledgeable person in your final answer — not stiff or robotic.
- If the message includes attached file or link content, use it directly — don't ask the user to repeat it.
- Don't use a tool unless you actually need it. Never call more tools than necessary.
- Only the JSON should appear in your reply when using a tool — no extra commentary.
"""

# ====================== IDENTITY LEAK PROTECTION ======================
IDENTITY_LEAK_PATTERNS = [
    (re.compile(r'\box[\s\-]?alpha\b', re.I), "JagX AI"),
    (re.compile(r'\bqwen(?:[\s\-]?\d(\.\d)?)?\b', re.I), "JagX AI"),
    (re.compile(r'\bllama[\s\-]?\d(\.\d)?\b', re.I), "JagX AI"),
    (re.compile(r'\bgroq\b', re.I), "JagX AI"),
    (re.compile(r'\bopenrouter\b', re.I), "JagX AI"),
    (re.compile(r'\bhugging\s?face\b', re.I), "JagX AI"),
    (re.compile(r'\bnvidia\b', re.I), "JagX AI"),
    (re.compile(r'\bdeepseek\b', re.I), "JagX AI"),
    (re.compile(r'\bmistral\b', re.I), "JagX AI"),
    (re.compile(r'\bgemini\b', re.I), "JagX AI"),
    (re.compile(r'\b(openai|chatgpt|gpt-?\d)\b', re.I), "JagX AI"),
    (re.compile(r'\b(anthropic|claude)\b', re.I), "JagX AI"),
    (re.compile(r'\bmeta\s?(ai|llama)\b', re.I), "JagX AI"),
    (re.compile(r'\bundisclosed (organi[sz]ation|company|provider|developer)\b', re.I), "JagX & JRILICENSE"),
    (re.compile(r'\bdeveloped by an? (third[\s\-]?party|stealth)[^.,\n]*', re.I), "developed by JagX & JRILICENSE"),
]

def sanitize_identity(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in IDENTITY_LEAK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

# ====================== INTENT ROUTING (fast chat vs tool/reasoning) ======================
CODE_KEYWORDS = ["code", "python", "javascript", "java ", "c++", "function", "debug", "script", "algorithm",
                 "compile", "run this", "execute", "bug", "error in my", "syntax", "programming"]
IMAGE_KEYWORDS = ["generate an image", "draw", "create an image", "picture of", "image of", "make me an image",
                  "illustrate", "generate a picture"]
DOC_KEYWORDS = ["pdf", " cv", "resume", "portfolio", "cover letter", ".zip", "generate a document",
                "generate a report", "write a report"]
JOB_KEYWORDS = ["job search", "find a job", "find jobs", "apply for a job", "job application", "remote job", "job listing"]
SEARCH_KEYWORDS = ["search for", "look up", "latest news", "current price", "what's happening", "who is the current",
                    "today's", "right now"]

def needs_tools(user_message: str, has_attachments: bool) -> bool:
    if has_attachments:
        return True
    text = (user_message or "").lower()
    if extract_urls(user_message):
        return True
    all_keywords = CODE_KEYWORDS + IMAGE_KEYWORDS + DOC_KEYWORDS + JOB_KEYWORDS + SEARCH_KEYWORDS
    return any(kw in text for kw in all_keywords)

# ====================== WATERMARK ======================
def add_invisible_watermark(text: str) -> str:
    ZWSP, ZWNJ, ZWJ = "\u200B", "\u200C", "\u200D"
    watermark = "".join([ZWNJ, ZWSP, ZWSP, ZWSP, ZWJ, ZWJ, ZWJ, ZWSP])
    if not text or len(text) < 15:
        return text + watermark
    third = len(text) // 3
    return text[:3] + watermark + text[3:third] + watermark + text[third:third*2] + watermark + text[third*2:]

# ====================== DATABASE LAYER (Neon Postgres, with file fallback) ======================
db_connection_pool = None
if DB_ENABLED:
    try:
        db_connection_pool = pg_pool.SimpleConnectionPool(1, 10, DATABASE_URL, sslmode="require")
    except Exception as e:
        logger.error(f"Could not create DB connection pool: {e}")
        DB_ENABLED = False

def get_conn():
    return db_connection_pool.getconn()

def put_conn(conn):
    db_connection_pool.putconn(conn)

def init_db():
    if not DB_ENABLED:
        logger.warning("DATABASE_URL not set — using local file storage. Data will NOT survive redeploys on Render's free tier.")
        return
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS api_keys (
                api_key TEXT PRIMARY KEY, owner TEXT, tier TEXT DEFAULT 'free',
                active BOOLEAN DEFAULT TRUE, created_at DOUBLE PRECISION
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS training_data (
                id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ DEFAULT now(),
                input TEXT, augmented_input TEXT, output TEXT, provider TEXT
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS job_drafts (
                draft_id TEXT PRIMARY KEY, subject TEXT, body TEXT,
                job_title TEXT, company TEXT, created_at DOUBLE PRECISION
            )""")
        conn.commit()
    finally:
        put_conn(conn)
    migrate_legacy_keys_file()

def migrate_legacy_keys_file():
    if not os.path.exists(KEYS_FILE):
        return
    try:
        with open(KEYS_FILE, "r") as f:
            legacy = json.load(f)
        if not legacy:
            return
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                for k, v in legacy.items():
                    cur.execute(
                        """INSERT INTO api_keys (api_key, owner, tier, active, created_at)
                           VALUES (%s,%s,%s,%s,%s) ON CONFLICT (api_key) DO NOTHING""",
                        (k, v.get("owner"), v.get("tier", "free"), v.get("active", True), v.get("created_at", time.time()))
                    )
            conn.commit()
            logger.info(f"Migrated {len(legacy)} legacy keys from keys.json into the database.")
        finally:
            put_conn(conn)
    except Exception as e:
        logger.warning(f"legacy key migration failed: {e}")

# ---- Keys: DB-backed with file fallback ----
def load_keys_file() -> dict:
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "w") as f:
            json.dump({}, f)
    with open(KEYS_FILE, "r") as f:
        return json.load(f)

def save_keys_file(keys: dict):
    tmp_path = KEYS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(keys, f, indent=2)
    os.replace(tmp_path, KEYS_FILE)

def db_get_key(key: str) -> Optional[dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM api_keys WHERE api_key=%s", (key,))
            return cur.fetchone()
    finally:
        put_conn(conn)

def db_create_key(new_key: str, owner: str, tier: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO api_keys (api_key, owner, tier, active, created_at) VALUES (%s,%s,%s,TRUE,%s)",
                        (new_key, owner, tier, time.time()))
        conn.commit()
    finally:
        put_conn(conn)

def db_list_keys() -> list:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM api_keys ORDER BY created_at DESC")
            return cur.fetchall()
    finally:
        put_conn(conn)

def db_set_key_active(key: str, active: bool) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE api_keys SET active=%s WHERE api_key=%s", (active, key))
            n = cur.rowcount
        conn.commit()
        return n > 0
    finally:
        put_conn(conn)

def db_set_key_tier(key: str, tier: str) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE api_keys SET tier=%s WHERE api_key=%s", (tier, key))
            n = cur.rowcount
        conn.commit()
        return n > 0
    finally:
        put_conn(conn)

def db_delete_key(key: str) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM api_keys WHERE api_key=%s", (key,))
            n = cur.rowcount
        conn.commit()
        return n > 0
    finally:
        put_conn(conn)

def db_rotate_key(old_key: str, new_key: str) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE api_keys SET api_key=%s, created_at=%s WHERE api_key=%s", (new_key, time.time(), old_key))
            n = cur.rowcount
        conn.commit()
        return n > 0
    finally:
        put_conn(conn)

def find_key_record(key: str) -> Tuple[Optional[dict], Optional[str]]:
    if not key:
        return None, None
    if key in PERMANENT_KEYS:
        return {"tier": "admin", "active": True}, "permanent"
    if DB_ENABLED:
        row = db_get_key(key)
        return (dict(row), "db") if row else (None, None)
    keys = load_keys_file()
    return (keys[key], "local") if key in keys else (None, None)

# ---- Training data: DB-backed with file fallback ----
def log_training_example(raw_input: str, augmented_input: str, output: str, provider: Optional[str]):
    if not TRAINING_DATA_ENABLED:
        return
    try:
        aug = augmented_input if augmented_input != raw_input else None
        if DB_ENABLED:
            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO training_data (input, augmented_input, output, provider) VALUES (%s,%s,%s,%s)",
                        (raw_input[:MAX_TRAINING_LOG_FIELD_LEN], aug[:MAX_TRAINING_LOG_FIELD_LEN] if aug else None,
                         output[:MAX_TRAINING_LOG_FIELD_LEN], provider)
                    )
                conn.commit()
            finally:
                put_conn(conn)
        else:
            record = {"timestamp": datetime.utcnow().isoformat(), "input": raw_input[:MAX_TRAINING_LOG_FIELD_LEN],
                      "augmented_input": aug[:MAX_TRAINING_LOG_FIELD_LEN] if aug else None,
                      "output": output[:MAX_TRAINING_LOG_FIELD_LEN], "provider": provider}
            with training_lock:
                with open(TRAINING_DATA_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"training data logging failed: {e}")

# ---- Job drafts: DB-backed with file fallback ----
def create_draft_record(draft_id: str, subject: str, body: str, job_title: str, company: str):
    if DB_ENABLED:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO job_drafts (draft_id, subject, body, job_title, company, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                            (draft_id, subject, body, job_title, company, time.time()))
            conn.commit()
        finally:
            put_conn(conn)
    else:
        with drafts_lock:
            drafts = _load_drafts_file()
            drafts = {k: v for k, v in drafts.items() if time.time() - v.get("created_at", 0) < DRAFT_E