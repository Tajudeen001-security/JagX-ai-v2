"""
JagX AI 6.2
General Purpose AI + Reasoning/Tool-Use + Vision + Image Gen + File/Link Reading
Created by JagX & JRILICENSE
"""

import os
import io
import csv
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

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jagx-ai")

# ====================== CONFIG ======================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY", "")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")

PISTON_URL = "https://emkc.org/api/v2/piston"

KEYS_FILE = "keys.json"
KNOWLEDGE_FILE = "jagx_knowledge.json"
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
MAX_IMAGE_BYTES = 8 * 1024 * 1024       # 8MB decoded, per image

MAX_FILES = 3
MAX_FILE_BYTES = 15 * 1024 * 1024       # 15MB decoded, per file
MAX_LINKS = 3
MAX_LINK_BYTES = 3 * 1024 * 1024        # 3MB downloaded, per link
MAX_EXTRACTED_TEXT_LEN = 12000          # chars fed into the model per file/link

APP_START_TIME = time.time()

app = FastAPI(
    title="JagX AI 6.2",
    description="General Purpose AI with reasoning, tools, vision, image gen, and file/link reading by JagX & JRILICENSE",
    version="6.2.0"
)

lock = threading.Lock()
rate_limit_store = defaultdict(list)
auth_attempt_store = defaultdict(list)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend domain once you have one
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

AGENT_SYSTEM_PROMPT = f"""You are JagX AI 6.2 — a powerful general-purpose AI created by JagX and JRILICENSE.

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
- Current date and time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

TOOLS — you may use these when you genuinely need current information or must execute/verify code.
To use a tool, reply with ONLY a raw JSON object, nothing else, in one of these forms:
{{"tool": "web_search", "input": "search query"}}
{{"tool": "run_code", "input": {{"language": "python", "code": "print('hello')"}}}}

When you are ready to answer the user, reply with ONLY:
{{"final": "your complete, well-formatted answer"}}

Rules:
- If the message includes attached file or link content, it will already be given to you as context — use it directly, don't ask the user to repeat it.
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
        headers = {"User-Agent": "Mozilla/5.0 (compatible; JagXAI/6.2)"}
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
    # Unknown extension — try plain text as a best-effort fallback
    return extract_text_from_plain(data)

def decode_and_check_file(filename: str, content_b64: str) -> bytes:
    approx_bytes = (len(content_b64) * 3) // 4
    if approx_bytes > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File '{filename}' too large (max {MAX_FILE_BYTES // (1024*1024)}MB).")
    try:
        return base64.b64decode(content_b64)
    except Exception:
        raise HTTPException(status_code=400, detail=f"File '{filename}' is not valid base64.")

# ====================== SANDBOXED CODE EXECUTION (Piston) ======================
_piston_runtimes_cache = {"data": None, "ts": 0}

def _get_piston_runtimes():
    now = time.time()
    if _piston_runtimes_cache["data"] and now - _piston_runtimes_cache["ts"] < 3600:
        return _piston_runtimes_cache["data"]
    try:
        r = HTTP.get(f"{PISTON_URL}/runtimes", timeout=10)
        if r.status_code == 200:
            _piston_runtimes_cache["data"] = r.json()
            _piston_runtimes_cache["ts"] = now
            return _piston_runtimes_cache["data"]
    except Exception as e:
        logger.warning(f"piston runtimes fetch failed: {e}")
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
        r = HTTP.post(f"{PISTON_URL}/execute", json=payload, timeout=20)
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
        logger.warning(f"code sandbox failed: {e}")
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
            payload = {"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.5}
            r = HTTP.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                content = _extract_openai_style_content(r.json())
                if content:
                    return content
            else:
                logger.warning(f"Groq returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"Groq call failed: {e}")
    if HF_TOKEN:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.5
            }
            r = HTTP.post(HF_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                content = _extract_openai_style_content(r.json())
                if content:
                    return content
        except Exception as e:
            logger.warning(f"HF call failed: {e}")
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
            r = HTTP.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=70)
            if r.status_code == 200:
                content = _extract_openai_style_content(r.json())
                if content:
                    return content
        except Exception as e:
            logger.warning(f"NVIDIA call failed: {e}")
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
            return raw
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

def build_augmented_message(
    user_message: str,
    image_descriptions: List[str],
    file_texts: List[Tuple[str, str]],
    link_texts: List[Tuple[str, str]]
) -> str:
    parts = [user_message]
    for desc in image_descriptions:
        parts.append(f"\n\n[Attached image content]\n{desc}")
    for fname, ftext in file_texts:
        parts.append(f"\n\n[Content of attached file: {fname}]\n{ftext}")
    for url, ltext in link_texts:
        parts.append(f"\n\n[Content of link: {url}]\n{ltext}")
    return "".join(parts)

def generate_response(
    user_message: str,
    history: Optional[List[Dict]] = None,