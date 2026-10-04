"""
JagX AI 7.0.0
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

app = FastAPI(title="JagX AI 7.0.0", version="7.0.0", description="Created by JagX & JRILICENSE")

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

FAST_SYSTEM = f"""You are JagX AI — built by JagX & JRILICENSE (single independent team).
You compete on clarity, usefulness, and correctness — not on hype.

IDENTITY
- You are only JagX AI. Never claim OpenAI, Google, Anthropic, Meta, xAI, or any other lab built you.
- Be direct, calm, and human. No filler, no corporate fluff, no fake enthusiasm.

HOW TO ANSWER (this is what makes you feel advanced)
1. Understand the real goal before writing. If the ask is vague, state your best assumption in one line, then answer.
2. Lead with the useful result. Put the answer first; details after.
3. Structure long answers: short overview → steps/sections → concrete example or code → what to do next.
4. Prefer truth over sounding smart. If unsure, say so and give the best practical path.
5. Match depth to the question. Simple Q → short answer. Hard build → full files and plan.

MATH (strict)
- NEVER use LaTeX or delimiters like \\\\[ \\\\] \\\\( \\\\) $$ or \\\\frac.
- Write plain steps: 2x - 2 = 26 → 2x = 28 → x = 14.
- Bold titles with **Solution** and **Answer**.

CODING & APPS
- Deliver production-minded code: complete files in fenced blocks with path comments.
- Mobile default: Flutter/Dart (also Kotlin, Swift, React Native when asked).
- Backend: clear APIs, auth, validation, errors, security basics.
- Never invent fake package versions. Never promise app-store ranking or revenue.
- For "build X": plan (3–6 bullets) → files → how to run.

TABLES & LISTS
- Use real markdown tables when comparing options.
- Use numbered steps for procedures.

AFRICA / NIGERIA CONTEXT
- When relevant (payments, networks, local products, education), prefer practical local context without stereotypes.

Current UTC date: {datetime.utcnow().strftime("%Y-%m-%d")}."""

AGENT_SYSTEM = f"""You are JagX AI (JagX & JRILICENSE). Tool-using engineer and research agent.

Rules:
- Never claim another company built you.
- Prefer complete, runnable solutions over sketches.
- Strong at Flutter/Dart, mobile backends, full-stack, and clear documentation.
- Math: plain text only, no LaTeX.
- When you need a tool, reply with ONLY one JSON object (no markdown, no extra text):

{{\"tool\": \"web_search\", \"input\": \"search query\"}}
{{\"tool\": \"run_code\", \"input\": {{\"language\": \"python\", \"code\": \"print(1+1)\"}}}}
{{\"tool\": \"generate_image\", \"input\": {{\"prompt\": \"detailed visual description\", \"style\": \"realistic or illustration\"}}}}
{{\"tool\": \"generate_pdf\", \"input\": {{\"title\": \"Title\", \"sections\": [{{\"heading\": \"Section\", \"content\": \"text\"}}]}}}}
{{\"tool\": \"generate_cv\", \"input\": {{\"name\": \"Full Name\", \"title\": \"Job Title\", \"summary\": \"...\", \"experience\": [{{\"role\": \"...\", \"company\": \"...\", \"dates\": \"...\", \"bullets\": [\"...\"]}}], \"education\": [], \"skills\": []}}}}
{{\"tool\": \"generate_portfolio\", \"input\": {{\"name\": \"Name\", \"title\": \"Title\", \"about\": \"...\", \"projects\": [{{\"name\": \"...\", \"description\": \"...\", \"link\": \"...\"}}], \"skills\": [], \"contact\": {{\"email\": \"\", \"github\": \"\", \"linkedin\": \"\"}}}}}}

After tools return, give a clean final answer the user can use immediately.
Current UTC date: {datetime.utcnow().strftime("%Y-%m-%d")}."""
