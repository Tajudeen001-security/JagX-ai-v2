"""
JagX file readers — screenshots/images, PDF, ZIP.
Free-first: pypdf + zipfile + HF/OpenRouter vision when keys exist.
Created by JagX & JRILICENSE.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import zipfile
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger("jagx-ai")

try:
    HTTP  # type: ignore  # noqa: F821
except NameError:
    HTTP = requests.Session()

MAX_UPLOAD_BYTES = int(os.environ.get("JAGX_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.environ.get("JAGX_MAX_PDF_PAGES", "40"))
MAX_ZIP_FILES = int(os.environ.get("JAGX_MAX_ZIP_FILES", "40"))
MAX_TEXT_OUT = 12000

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

HF_VISION_MODELS = [
    os.environ.get("JAGX_HF_VISION_MODEL", "").strip(),
    "Salesforce/blip-image-captioning-large",
    "nlpconnect/vit-gpt2-image-captioning",
]
HF_VISION_MODELS = [m for m in HF_VISION_MODELS if m]

OPENROUTER_VISION_MODEL = os.environ.get(
    "JAGX_OPENROUTER_VISION_MODEL",
    "google/gemini-2.0-flash-exp:free",
)


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", (name or "file")[:120])


def read_pdf_bytes(data: bytes, filename: str = "doc.pdf") -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            return (
                "PDF support needs the pypdf package. "
                "Add pypdf to requirements and redeploy. "
                f"Received file: {_safe_name(filename)} ({len(data)} bytes)."
            )
    try:
        reader = PdfReader(io.BytesIO(data))
        n = len(reader.pages)
        parts = [f"**PDF:** {_safe_name(filename)}", f"**Pages:** {n}"]
        texts = []
        for i, page in enumerate(reader.pages[:MAX_PDF_PAGES]):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            t = t.strip()
            if t:
                texts.append(f"--- Page {i + 1} ---\n{t}")
        body = "\n\n".join(texts) if texts else "(No extractable text — may be scanned images only. Upload a screenshot of a page or use an OCR app first.)"
        out = "\n".join(parts) + "\n\n" + body
        return out[:MAX_TEXT_OUT]
    except Exception as e:
        return f"Could not read PDF {_safe_name(filename)}: {e}"


def read_zip_bytes(data: bytes, filename: str = "archive.zip") -> str:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as e:
        return f"Could not open ZIP {_safe_name(filename)}: {e}"
    names = zf.namelist()
    lines = [
        f"**ZIP:** {_safe_name(filename)}",
        f"**Files inside:** {len(names)}",
        "",
        "**Contents:**",
    ]
    text_ext = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".html", ".css", ".xml", ".yml", ".yaml", ".toml", ".ini", ".log", ".dart", ".java", ".kt", ".go", ".rs", ".c", ".cpp", ".h", ".sql", ".sh"}
    shown = 0
    for name in names[:MAX_ZIP_FILES]:
        info = zf.getinfo(name)
        if name.endswith("/"):
            lines.append(f"- 📁 {name}")
            continue
        size = info.file_size
        lines.append(f"- 📄 {name} ({size} bytes)")
        ext = os.path.splitext(name)[1].lower()
        if ext in text_ext and size < 200_000 and shown < 12:
            try:
                raw = zf.read(name)
                if b"\x00" in raw[:1000]:
                    continue
                text = raw.decode("utf-8", errors="replace")
                snippet = text[:1500]
                lines.append(f"  ```\n  {snippet}\n  ```")
                shown += 1
            except Exception:
                pass
        elif ext == ".pdf" and size < 5_000_000 and shown < 8:
            try:
                pdf_txt = read_pdf_bytes(zf.read(name), name)
                lines.append(pdf_txt[:2000])
                shown += 1
            except Exception:
                pass
    if len(names) > MAX_ZIP_FILES:
        lines.append(f"... and {len(names) - MAX_ZIP_FILES} more entries not listed.")
    return "\n".join(lines)[:MAX_TEXT_OUT]


def _caption_hf(image_bytes: bytes) -> Optional[str]:
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for model in HF_VISION_MODELS:
        try:
            url = f"https://api-inference.huggingface.co/models/{model}"
            r = HTTP.post(url, headers=headers, data=image_bytes, timeout=45)
            if r.status_code == 503:
                continue
            if r.status_code != 200:
                logger.warning("HF vision %s: %s %s", model, r.status_code, r.text[:200])
                continue
            data = r.json()
            if isinstance(data, list) and data:
                cap = data[0].get("generated_text") or data[0].get("caption") or str(data[0])
                return str(cap)
            if isinstance(data, dict):
                return str(data.get("generated_text") or data.get("caption") or data)
            return str(data)
        except Exception as e:
            logger.warning("HF vision error %s: %s", model, e)
    return None


def _caption_openrouter(image_bytes: bytes, prompt: str) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        return None
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        mime = "image/jpeg"
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        elif image_bytes[:4] == b"RIFF":
            mime = "image/webp"
        payload = {
            "model": OPENROUTER_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": 800,
        }
        r = HTTP.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if r.status_code != 200:
            logger.warning("OpenRouter vision: %s %s", r.status_code, r.text[:240])
            return None
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("OpenRouter vision error: %s", e)
        return None


def read_image_bytes(data: bytes, filename: str = "image.png", question: str = "") -> str:
    if len(data) > MAX_UPLOAD_BYTES:
        return f"Image too large ({len(data)} bytes). Max is {MAX_UPLOAD_BYTES}."
    q = (question or "").strip() or (
        "Describe this image or screenshot in detail. "
        "If it contains text (UI, chat, error, homework), transcribe the important text clearly. "
        "If it is a diagram, explain what it shows."
    )
    cap = _caption_openrouter(data, q)
    if not cap:
        cap = _caption_hf(data)
    meta = f"**Image:** {_safe_name(filename)} ({len(data)} bytes)"
    if cap:
        return f"{meta}\n\n**What I see:**\n{cap}"[:MAX_TEXT_OUT]
    return (
        f"{meta}\n\n"
        "Could not run vision on this image right now. "
        "Ensure HF_TOKEN or OPENROUTER_API_KEY is set on Render, or paste the text from the screenshot."
    )


def read_plain_text(data: bytes, filename: str = "file.txt") -> str:
    try:
        text = data.decode("utf-8")
    except Exception:
        text = data.decode("latin-1", errors="replace")
    return f"**File:** {_safe_name(filename)}\n\n{text[:MAX_TEXT_OUT]}"


def read_any_file(data: bytes, filename: str, content_type: str = "", question: str = "") -> str:
    name = (filename or "upload").lower()
    ctype = (content_type or "").lower()
    if len(data) > MAX_UPLOAD_BYTES:
        return f"File too large ({len(data)} bytes). Max {MAX_UPLOAD_BYTES} bytes."

    if name.endswith(".pdf") or "pdf" in ctype:
        return read_pdf_bytes(data, filename)
    if name.endswith(".zip") or "zip" in ctype:
        return read_zip_bytes(data, filename)
    if any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")) or ctype.startswith("image/"):
        return read_image_bytes(data, filename, question=question)
    if any(name.endswith(ext) for ext in (".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".dart", ".html", ".css", ".xml", ".log")):
        return read_plain_text(data, filename)
    if data[:2] == b"PK":
        return read_zip_bytes(data, filename)
    if data[:4] == b"%PDF":
        return read_pdf_bytes(data, filename)
    return (
        f"Received **{_safe_name(filename)}** ({len(data)} bytes, type={ctype or 'unknown'}). "
        "Supported: images/screenshots (png/jpg/webp), PDF, ZIP, and common text files."
    )


def register_file_routes():
    app = globals().get("app")
    if app is None:
        logger.warning("jagx_files: no app")
        return

    try:
        from fastapi import File, Form, UploadFile
    except Exception:
        try:
            from fastapi import File, Form, UploadFile  # typo guard
        except Exception:
            pass
        try:
            from fastapi import File, Form, UploadFile
        except Exception as e:
            logger.warning("jagx_files: UploadFile missing: %s", e)
            return

    from fastapi import File, Form, UploadFile

    @app.post("/read_file")
    async def read_file_endpoint(
        file: UploadFile = File(...),
        question: str = Form(""),
    ):
        data = await file.read()
        result = read_any_file(data, file.filename or "upload", file.content_type or "", question=question)
        return {"filename": file.filename, "size": len(data), "analysis": result}

    @app.post("/read_image")
    async def read_image_endpoint(file: UploadFile = File(...), question: str = Form("")):
        data = await file.read()
        return {"analysis": read_image_bytes(data, file.filename or "image.png", question=question)}

    @app.post("/read_pdf")
    async def read_pdf_endpoint(file: UploadFile = File(...)):
        data = await file.read()
        return {"analysis": read_pdf_bytes(data, file.filename or "doc.pdf")}

    @app.post("/read_zip")
    async def read_zip_endpoint(file: UploadFile = File(...)):
        data = await file.read()
        return {"analysis": read_zip_bytes(data, file.filename or "archive.zip")}

    logger.info("JagX file routes: /read_file /read_image /read_pdf /read_zip")


try:
    register_file_routes()
except Exception as e:
    logger.warning("register_file_routes: %s", e)
