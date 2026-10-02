"""
JagX AI 6.9.3
Full Features Edition
- Fast normal chat; slower only for multi-job / coding tools
- Local GitHub knowledge brain (works with zero cloud LLMs)
- Provider order: Groq (fast) → OpenRouter → HF (last)
- Tools: web_search, run_code, generate_image, pdf, cv, portfolio
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
import zipfile
import logging
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from urllib.parse import quote
from datetime import datetime
from io import BytesIO

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
from fpdf import FPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jagx-ai")

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
MAX_CODE_LEN = 12000
MAX_TOOL_RESULT_LEN = 3000
MAX_BODY_BYTES = 15 * 1024 * 1024
GLOBAL_IP_RPM = 100
DRAFT_EXPIRY_SECONDS = 48 * 3600
APP_START_TIME = time.time()

app = FastAPI(title="JagX AI 6.9.3", version="6.9.3", description="Created by JagX & JRILICENSE")

lock = threading.Lock()
drafts_lock = threading.Lock()
training_lock = threading.Lock()
rate_limit_store = defaultdict(list)
global_ip_rate_store = defaultdict(list)

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

LOCAL_KB = []

def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s\+\-\*\?\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def load_local_brain():
    global LOCAL_KB
    entries = []
    for fname in ["jagx_knowledge.json"]:
        if os.path.exists(fname):
            try:
                data = json.load(open(fname))
                if isinstance(data, list):
                    for row in data:
                        q = row.get("question") or row.get("q") or ""
                        a = row.get("answer") or row.get("a") or ""
                        if q and a:
                            entries.append({"keys": [_normalize(q)], "answer": a})
            except Exception as e:
                logger.warning(f"load {fname}: {e}")
    for name in sorted(os.listdir(".")):
        if name.startswith("knowledge_pack") and name.endswith(".json"):
            try:
                data = json.load(open(name))
                if isinstance(data, list):
                    for row in data:
                        keys = row.get("k") or row.get("keys") or []
                        a = row.get("a") or row.get("answer") or ""
                        if isinstance(keys, str):
                            keys = [keys]
                        keys = [_normalize(k) for k in keys if k]
                        if keys and a:
                            entries.append({"keys": keys, "answer": a})
            except Exception as e:
                logger.warning(f"load {name}: {e}")
    LOCAL_KB = entries
    logger.info(f"Local brain loaded: {len(LOCAL_KB)} entries")

def match_local(user_message: str) -> Optional[str]:
    if not LOCAL_KB:
        return None
    q = _normalize(user_message)
    if not q or len(q) > 200:
        return None
    best = None
    best_score = 0.0
    q_tokens = set(q.split())
    for entry in LOCAL_KB:
        for k in entry["keys"]:
            if not k:
                continue
            if q == k:
                return entry["answer"]
            if k in q or q in k:
                score = len(k) / max(len(q), 1)
                if score > best_score:
                    best_score = score
                    best = entry["answer"]
                continue
            k_tokens = set(k.split())
            if not k_tokens:
                continue
            inter = len(q_tokens & k_tokens)
            union = len(q_tokens | k_tokens) or 1
            jacc = inter / union
            if inter >= 2 and jacc > best_score:
                best_score = jacc
                best = entry["answer"]
    if best and best_score >= 0.45:
        return best
    return None

load_local_brain()

FAST_SYSTEM = f"""You are JagX AI, created by JagX and JRILICENSE. Be helpful, clear and natural.
Never mention OpenAI, Groq, Meta, Google, Anthropic or any other company. You are only JagX AI by JagX & JRILICENSE.

CODING & MOBILE APPS (when the user asks to build software):
- Prefer complete, production-oriented code over sketches.
- Mobile: Flutter/Dart first for cross-platform; also React Native, Kotlin, Swift when asked.
- Structure answers: short plan → full files in markdown code fences with path comments → how to run/build.
- Backend for ranked apps: auth (JWT/OAuth), rate limits, validation, logging, clean API design, caching, security basics.
- UI: accessibility, loading/error states, offline-friendly patterns when relevant.
- Do not invent fake package versions; use stable common APIs.
- Never promise store ranking or revenue. Explain quality factors (performance, UX, reviews, ASO) honestly.

IMAGE REQUESTS: If the user wants an image and tools are available, the agent path handles generation. Otherwise describe a clear image prompt they can use.

Current date: {datetime.utcnow().strftime("%Y-%m-%d")}."""

AGENT_SYSTEM = f"""You are JagX AI, created by JagX and JRILICENSE.
Never claim to be made by any other company.

You are strong at software engineering: Flutter/Dart mobile apps, React Native, Android/iOS, Node/Python/FastAPI backends, auth, APIs, and app-store-ready structure (not ranking guarantees).

When you need a tool, reply with ONLY a JSON object (no extra text):
{{"tool": "web_search", "input": "search query"}}
{{"tool": "run_code", "input": {{"language": "python", "code": "print(1+1)"}}}}
{{"tool": "generate_image", "input": {{"prompt": "detailed visual description", "style": "realistic or illustration"}}}}
{{"tool": "generate_pdf", "input": {{"title": "Title", "sections": [{{"heading": "Section", "content": "text"}}]}}}}
{{"tool": "generate_cv", "input": {{"name": "Full Name", "title": "Job Title", "summary": "...", "experience": [{{"role": "...", "company": "...", "dates": "...", "bullets": ["..."]}}], "education": [], "skills": []}}}}
{{"tool": "generate_portfolio", "input": {{"name": "Name", "title": "Title", "about": "...", "projects": [{{"name": "...", "description": "...", "link": "..."}}], "skills": [], "contact": {{"email": "", "github": "", "linkedin": ""}}}}}}

For coding tasks prefer a final answer with complete files. Use run_code to verify small Python snippets when helpful.
For image requests always use generate_image with a detailed prompt.

When ready to answer the user, reply with ONLY:
{{"final": "your complete answer here"}}

Current date: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
"""

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

def free_web_search(query: str, max_results: int = 5) -> str:
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = HTTP.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return "Search currently unavailable."
        results = []
        blocks = re.findall(r'class="result__a"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td)>', r.text, re.DOTALL)
        for title, snippet in blocks[:max_results]:
            title = re.sub(r'<.*?>', '', title).strip()
            snippet = re.sub(r'<.*?>', '', snippet).strip()
            if title and len(snippet) > 30:
                results.append(f"**{title}**\n{snippet}")
        if not results:
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|td)>', r.text, re.DOTALL)
            for s in snippets[:max_results]:
                clean = re.sub(r'<.*?>', '', s).strip()
                if len(clean) > 40:
                    results.append(clean)
        if results:
            return "Here’s what I found online:\n\n" + "\n\n".join(results)
        return "No relevant results found."
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return "Search failed at the moment."

def run_code_sandboxed(language: str, code: str) -> str:
    if len(code) > MAX_CODE_LEN:
        return "Code is too long."
    try:
        payload = {"language": language.lower(), "version": "*", "files": [{"content": code}]}
        r = HTTP.post(f"{PISTON_URL}/execute", json=payload, timeout=15)
        if r.status_code != 200:
            return f"Execution failed (status {r.status_code})"
        data = r.json()
        run = data.get("run", {})
        out = (run.get("stdout") or "") + (run.get("stderr") or "")
        return (out[:MAX_TOOL_RESULT_LEN] or "No output").strip()
    except Exception as e:
        return f"Sandbox error: {str(e)[:120]}"

def _sanitize(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")

def generate_pdf_document(title: str, sections: List[Dict]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _sanitize(title))
    pdf.ln(3)
    for sec in sections[:25]:
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
    pdf.ln(3)
    if cv.get("summary"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "SUMMARY", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize(cv["summary"]))
        pdf.ln(2)
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
            pdf.ln(1)
    if cv.get("skills"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "SKILLS", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize(", ".join(cv["skills"])))
    return bytes(pdf.output())

def generate_portfolio_html(data: dict) -> str:
    name = data.get("name", "My Portfolio")
    title = data.get("title", "")
    about = data.get("about", "")
    projects = data.get("projects", [])
    skills = data.get("skills", [])
    contact = data.get("contact", {})
    projects_html = ""
    for p in projects:
        link = f'<p><a href="{p.get("link")}" target="_blank">View Project</a></p>' if p.get("link") else ""
        projects_html += f"""
        <div style="border:1px solid #ddd; padding:16px; margin:12px 0; border-radius:8px;">
            <h3>{p.get('name', 'Project')}</h3>
            <p>{p.get('description', '')}</p>
            {link}
        </div>"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name} - Portfolio</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #222; }}
h1 {{ margin-bottom: 4px; }}
.title {{ color: #555; margin-top: 0; }}
a {{ color: #0066cc; }}
</style>
</head>
<body>
<h1>{name}</h1>
<p class="title">{title}</p>
<h2>About</h2>
<p>{about}</p>
<h2>Projects</h2>
{projects_html}
<h2>Skills</h2>
<p>{', '.join(skills) if skills else 'Not specified'}</p>
<h2>Contact</h2>
<p>{contact.get('email', '')}<br>{contact.get('github', '')}<br>{contact.get('linkedin', '')}</p>
<hr>
<small>Generated by JagX AI</small>
</body>
</html>"""

def create_zip(files: Dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return buffer.getvalue()

def call_llm(messages: list, max_tokens: int = 1100) -> Optional[str]:
    """Fast path first (Groq), then OpenRouter, then HF. Any one is enough."""
    if GROQ_API_KEY:
        try:
            r = HTTP.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.5},
                timeout=18,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            logger.warning(f"Groq status {r.status_code}")
        except Exception as e:
            logger.warning(f"Groq failed: {e}")
    if OPENROUTER_API_KEY:
        try:
            r = HTTP.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={"model": OPENROUTER_MODEL, "messages": messages, "max_tokens": max_tokens},
                timeout=22,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            logger.warning(f"OpenRouter status {r.status_code}")
        except Exception as e:
            logger.warning(f"OpenRouter failed: {e}")
    if HF_TOKEN:
        HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        try:
            r = HTTP.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                json={"model": HF_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.5},
                timeout=28,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            logger.warning(f"HF status {r.status_code}")
        except Exception as e:
            logger.warning(f"HF failed: {e}")
    return None

def needs_tools(msg: str) -> bool:
    text = (msg or "").lower()
    keywords = ["code", "python", "javascript", "typescript", "flutter", "dart", "kotlin", "swift",
                "react native", "android", "ios", "mobile app", "backend", "api", "fastapi",
                "run this", "execute", "pdf", "cv", "resume", "portfolio",
                "image", "picture", "draw", "logo", "illustration", "generate image",
                "search for", "look up", "latest", "job", "generate a", "create a", "build me", "write a"]
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
        for m in history[-4:]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    attachment = None
    for _ in range(MAX_TOOL_STEPS):
        raw = call_llm(messages, max_tokens=800)
        if not raw:
            break
        data = extract_json(raw)
        if not data:
            return sanitize_identity(raw), None
        if "final" in data:
            return sanitize_identity(str(data["final"])), attachment
        tool = data.get("tool")
        inp = data.get("input")
        if tool == "web_search":
            result = free_web_search(str(inp))
        elif tool == "run_code" and isinstance(inp, dict):
            result = run_code_sandboxed(inp.get("language", "python"), inp.get("code", ""))
        elif tool == "generate_image" and isinstance(inp, dict):
            try:
                prompt = (inp.get("prompt") or inp.get("description") or "abstract art").strip()
                style = (inp.get("style") or "").strip()
                full = (prompt + (", " + style if style else "")).strip()
                url = "https://image.pollinations.ai/prompt/" + quote(full) + "?width=1024&height=1024&nologo=true"
                attachment = {"type": "image", "url": url, "prompt": full, "filename": "jagx-image.png"}
                result = f"Image generated.\nURL: {url}\nPrompt used: {full}"
            except Exception as e:
                result = f"Image generation failed: {e}"
        elif tool == "generate_pdf" and isinstance(inp, dict):
            try:
                pdf_bytes = generate_pdf_document(inp.get("title", "Document"), inp.get("sections", []))
                attachment = {"type": "pdf", "filename": "document.pdf", "content_base64": base64.b64encode(pdf_bytes).decode()}
                result = "PDF generated successfully and attached."
            except Exception as e:
                result = f"PDF generation failed: {e}"
        elif tool == "generate_cv" and isinstance(inp, dict):
            try:
                pdf_bytes = generate_cv_pdf(inp)
                attachment = {"type": "pdf", "filename": "cv.pdf", "content_base64": base64.b64encode(pdf_bytes).decode()}
                result = "CV generated successfully and attached."
            except Exception as e:
                result = f"CV generation failed: {e}"
        elif tool == "generate_portfolio" and isinstance(inp, dict):
            try:
                html = generate_portfolio_html(inp)
                attachment = {"type": "html", "filename": "portfolio.html", "content_base64": base64.b64encode(html.encode()).decode()}
                result = "Portfolio generated successfully and attached."
            except Exception as e:
                result = f"Portfolio generation failed: {e}"
        else:
            result = "Unknown or unsupported tool."
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Tool result:\n{result}\n\nContinue."})
    fallback = call_llm([{"role": "system", "content": FAST_SYSTEM}, {"role": "user", "content": user_message}])
    return sanitize_identity(fallback or "I couldn't complete the request."), attachment

def generate_response(user_message: str, history: Optional[List] = None) -> Tuple[str, Optional[dict]]:
    local = match_local(user_message)
    if local and not needs_tools(user_message):
        return sanitize_identity(local), None
    if needs_tools(user_message):
        text, att = run_agent(user_message, history)
        if text:
            return text, att
        if local:
            return sanitize_identity(local), None
    messages = [{"role": "system", "content": FAST_SYSTEM}]
    if history:
        for m in history[-6:]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    result = call_llm(messages, max_tokens=700)
    if result:
        return sanitize_identity(result), None
    if local:
        return sanitize_identity(local), None
    return (
        sanitize_identity(
            "I am JagX AI. Live model links are busy, but I am still online. "
            "I can use local knowledge, run code, search the web, generate images, and build PDFs. "
            "Ask again, or say: write code, search, generate image, or make a PDF."
        ),
        None,
    )

def log_training(input_text: str, output_text: str):
    if not TRAINING_DATA_ENABLED:
        return
    try:
        with training_lock:
            with open(TRAINING_DATA_FILE, "a") as f:
                f.write(json.dumps({"input": input_text[:4000], "output": output_text[:8000], "ts": time.time()}) + "\n")
    except Exception:
        pass

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = None

@app.get("/health")
def health():
    return {"status": "ok", "service": "jagx-backend-v1", "version": "6.9.3", "uptime_seconds": int(time.time() - APP_START_TIME), "local_brain": len(LOCAL_KB), "providers": {"hf": bool(HF_TOKEN), "groq": bool(GROQ_API_KEY), "openrouter": bool(OPENROUTER_API_KEY)}}

@app.post("/chat")
def chat(req: ChatRequest, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    key = (x_api_key or "").strip()
    if not is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok, info = check_rate_limit(key)
    if not ok:
        raise HTTPException(status_code=429, detail=info)
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")
    text, attachment = generate_response(msg, req.history)
    text = add_invisible_watermark(text or "")
    log_training(msg, text)
    out = {"response": text, "rate_limit": info}
    if attachment:
        out["attachment"] = attachment
    return out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
