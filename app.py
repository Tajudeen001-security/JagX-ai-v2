"""
JagX AI 6.4
General Purpose AI + Reasoning/Tool-Use + Vision + Image Gen + File/Link Reading
+ PDF/CV Generation + Multi-Provider LLM Cascade + Training Data Collection
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
TRAINING_DATA_ENABLED = os.environ.get("JAGX_TRAINING_DATA_ENABLED", "true").lower() == "true"

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
MAX_TRAINING_LOG_FIELD_LEN = 6000

APP_START_TIME = time.time()

app = FastAPI(
    title="JagX AI 6.4",
    description="General Purpose AI with reasoning, tools, vision, image gen, file/link reading, PDF/CV generation, and a multi-provider LLM cascade by JagX & JRILICENSE",
    version="6.4.0"
)

lock = threading.Lock()
training_lock = threading.Lock()
rate_limit_store = defaultdict(list)
auth_attempt_store = defaultdict(list)

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
    retries = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

HTTP = _build_session()

AGENT_SYSTEM_PROMPT = f"""You are JagX AI 6.4 — a powerful general-purpose AI created by JagX and JRILICENSE.

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
- Reading attached files (PDF, Word, txt, csv) and links pasted by the user
- Generating downloadable PDF documents and professional CVs/resumes
- Current date and time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

TOOLS — you may use these when you genuinely need current information, must execute/verify code, or the user wants a document produced.
To use a tool, reply with ONLY a raw JSON object, nothing else, in one of these forms:
{{"tool": "web_search", "input": "search query"}}
{{"tool": "run_code", "input": {{"language": "python", "code": "print('hello')"}}}}
{{"tool": "generate_pdf", "input": {{"title": "Document Title", "sections": [{{"heading": "Section 1", "content": "Text here. Use lines starting with '- ' for bullet points."}}]}}}}
{{"tool": "generate_cv", "input": {{"name": "Full Name", "title": "Professional Title", "contact": {{"email": "...", "phone": "...", "location": "...", "linkedin": "..."}}, "summary": "2-3 sentence professional summary", "experience": [{{"role": "Job Title", "company": "Company", "dates": "2022 - Present", "bullets": ["Achievement 1", "Achievement 2"]}}], "education": [{{"degree": "Degree", "school": "School", "dates": "2018 - 2022"}}], "skills": ["Skill 1", "Skill 2"]}}}}

When you are ready to answer the user, reply with ONLY:
{{"final": "your complete, well-formatted answer"}}

Rules:
- If the message includes attached file or link content, it will already be given to you as context — use it directly, don't ask the user to repeat it.
- When a user asks for a document, report, or letter, use generate_pdf. When a user asks for a CV, resume, or asks you to turn their background into one, use generate_cv and infer/organize the fields from what they told you.
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
    tmp_path = KEYS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(keys, f, indent=2)
    os.replace(tmp_path, KEYS_FILE)

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

# ====================== SSRF-SAFE LINK READING ======================
def _is_public_ip(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            ip_str = info[4][0]
            ip = ipaddress.ip_address(ip_str)
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                return False
        return True
    except Exception:
        return False

def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        if host.lower() in ("localhost",) or host.startswith("169.254."):
            return False
        return _is_public_ip(host)
    except Exception:
        return False

def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_link_content(url: str) -> str:
    if not is_safe_url(url):
        return f"Error: '{url}' could not be read (blocked or invalid address)."
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; JagXAI/6.4)"}
        r = HTTP.get(url, headers=headers, timeout=15, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return f"Error: link returned status {r.status_code}."
        content_type = r.headers.get("Content-Type", "")
        raw = b""
        for chunk in r.iter_content(chunk_size=65536):
            raw += chunk
            if len(raw) > MAX_LINK_BYTES:
                break
        if len(raw) > MAX_LINK_BYTES:
            raw = raw[:MAX_LINK_BYTES]
        text = raw.decode(errors="ignore")
        if "html" in content_type or text.lstrip().startswith("<"):
            text = extract_readable_text(text)
        return text[:MAX_EXTRACTED_TEXT_LEN]
    except Exception as e:
        logger.warning(f"link fetch failed: {e}")
        return f"Error: could not read link ({str(e)[:150]})."

def extract_urls(text: str) -> List[str]:
    urls = re.findall(r'https?://[^\s<>"\'\)]+', text or "")
    seen = []
    for u in urls:
        if u not in seen:
            seen.append(u)
    return seen[:MAX_LINKS]

# ====================== FILE READING ======================
def extract_text_from_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages[:50]:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)[:MAX_EXTRACTED_TEXT_LEN]
    except Exception as e:
        return f"Error: could not read PDF ({str(e)[:150]})."

def extract_text_from_docx(data: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        return "\n".join(parts)[:MAX_EXTRACTED_TEXT_LEN]
    except Exception as e:
        return f"Error: could not read DOCX ({str(e)[:150]})."

def extract_text_from_plain(data: bytes) -> str:
    try:
        return data.decode(errors="ignore")[:MAX_EXTRACTED_TEXT_LEN]
    except Exception as e:
        return f"Error: could not read file ({str(e)[:150]})."

def extract_text_from_file(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    if name.endswith((".txt", ".csv", ".md", ".json", ".log")):
        return extract_text_from_plain(data)
    return extract_text_from_plain(data)

def decode_and_check_file(filename: str, content_b64: str) -> bytes:
    approx_bytes = (len(content_b64) * 3) // 4
    if approx_bytes > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File '{filename}' too large (max {MAX_FILE_BYTES // (1024*1024)}MB).")
    try:
        return base64.b64decode(content_b64)
    except Exception:
        raise HTTPException(status_code=400, detail=f"File '{filename}' is not valid base64.")

# ====================== PDF / CV GENERATION ======================
def slugify_filename(text: str, ext: str) -> str:
    base = re.sub(r'[^a-zA-Z0-9\-_]+', '_', (text or "").strip()).strip('_')[:60]
    return f"{base or 'document'}.{ext}"

def _sanitize_pdf_text(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")

def generate_pdf_document(title: str, sections: List[Dict[str, str]], author: str = "JagX AI") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, _sanitize_pdf_text(title))
    pdf.ln(1)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, _sanitize_pdf_text(f"Generated by {author} — {datetime.utcnow().strftime('%Y-%m-%d')}"))
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for sec in sections[:MAX_PDF_SECTIONS]:
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        if heading:
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, _sanitize_pdf_text(heading))
            pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        for line in content.split("\n"):
            line = line.rstrip()
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")):
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(0, 6, _sanitize_pdf_text("•  " + stripped[2:]))
            else:
                pdf.multi_cell(0, 6, _sanitize_pdf_text(line))
        pdf.ln(3)

    raw = pdf.output()
    return bytes(raw)

def build_pdf_tool_result(title: str, sections_input) -> Tuple[str, Optional[dict]]:
    if isinstance(sections_input, list):
        sections = sections_input
    else:
        sections = [{"heading": "", "content": str(sections_input)}]
    total_len = sum(len(s.get("content", "")) for s in sections if isinstance(s, dict))
    if total_len > MAX_PDF_CONTENT_LEN or len(sections) > MAX_PDF_SECTIONS:
        return "Error: requested PDF content is too large to generate.", None
    try:
        pdf_bytes = generate_pdf_document(title or "Document", sections)
        filename = slugify_filename(title or "document", "pdf")
        b64 = base64.b64encode(pdf_bytes).decode()
        attachment = {"type": "pdf", "filename": filename, "content_base64": b64}
        return f"PDF '{filename}' generated successfully and attached for the user to download.", attachment
    except Exception as e:
        logger.warning(f"pdf generation failed: {e}")
        return f"Error: PDF generation failed ({str(e)[:150]}).", None

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

    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, _sanitize_pdf_text(name), ln=True)

    if title:
        pdf.set_font("Helvetica", "", 13)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 8, _sanitize_pdf_text(title), ln=True)
        pdf.set_text_color(0, 0, 0)

    contact_line = "  |  ".join(
        v for v in [contact.get("email"), contact.get("phone"), contact.get("location"), contact.get("linkedin")] if v
    )
    if contact_line:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(0, 6, _sanitize_pdf_text(contact_line))
        pdf.set_text_color(0, 0, 0)

    pdf.ln(2)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), 210 - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    def section_header(text):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, _sanitize_pdf_text(text.upper()), ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    if summary:
        section_header("Summary")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize_pdf_text(summary))
        pdf.ln(3)

    if experience:
        section_header("Experience")
        for job in experience:
            role = job.get("role", "")
            company = job.get("company", "")
            dates = job.get("dates", "")
            bullets = job.get("bullets", []) or []
            pdf.set_font("Helvetica", "B", 11)
            header_line = f"{role} — {company}" if company else role
            pdf.cell(0, 6, _sanitize_pdf_text(header_line), ln=True)
            if dates:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(110, 110, 110)
                pdf.cell(0, 5, _sanitize_pdf_text(dates), ln=True)
                pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10.5)
            for b in bullets:
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(0, 5.5, _sanitize_pdf_text("•  " + str(b)))
            pdf.ln(2)

    if education:
        section_header("Education")
        for edu in education:
            degree = edu.get("degree", "")
            school = edu.get("school", "")
            dates = edu.get("dates", "")
            pdf.set_font("Helvetica", "B", 11)
            line = f"{degree} — {school}" if school else degree
            pdf.cell(0, 6, _sanitize_pdf_text(line), ln=True)
            if dates:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(110, 110, 110)
                pdf.cell(0, 5, _sanitize_pdf_text(dates), ln=True)
                pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

    if skills:
        section_header("Skills")
        pdf.set_font("Helvetica", "", 10.5)
        pdf.multi_cell(0, 6, _sanitize_pdf_text(", ".join(str(s) for s in skills)))

    raw = pdf.output()
    return bytes(raw)

def build_cv_tool_result(cv_input: dict) -> Tuple[str, Optional[dict]]:
    if not isinstance(cv_input, dict) or not cv_input.get("name"):
        return "Error: CV generation needs at least a name field.", None
    try:
        pdf_bytes = generate_cv_pdf(cv_input)
        filename = slugify_filename(f"{cv_input.get('name', 'cv')}_CV", "pdf")
        b64 = base64.b64encode(pdf_bytes).decode()
        attachment = {"type": "pdf", "filename": filename, "content_base64": b64}
        return f"CV '{filename}' generated successfully and attached for the user to download.", attachment
    except Exception as e:
        logger.warning(f"cv generation failed: {e}")
        return f"Error: CV generation failed ({str(e)[:150]}).", None

# ====================== SANDBOXED CODE EXECUTION (Piston) ======================
_piston_runtimes_cache = {"data": None, "