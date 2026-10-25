"""Built-in tools: web search, code sandbox, calc, PDF, CV, portfolio, fetch_url."""

from __future__ import annotations

import ast
import base64
import html as html_mod
import json
import logging
import operator
import re
from html import escape
from typing import Any, Optional
from urllib.parse import quote, urlparse, parse_qs

from jagx_api.config import settings
from jagx_api.providers import HTTP

logger = logging.getLogger("jagx-ai")

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def free_web_search(query: str, max_results: int = 6) -> str:
    q = (query or "").strip()[:220]
    if not q:
        return "empty query"
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(q)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        r = HTTP.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return _ddg_instant(q, max_results)
        results = []
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td)>',
            re.DOTALL,
        )
        for href, title, snippet in pattern.findall(r.text)[:max_results]:
            title = html_mod.unescape(re.sub(r"<.*?>", "", title)).strip()
            snippet = html_mod.unescape(re.sub(r"<.*?>", "", snippet)).strip()
            href = html_mod.unescape(href).strip()
            if href.startswith("//"):
                href = "https:" + href
            parsed = urlparse(href)
            if "duckduckgo.com" in (parsed.netloc or ""):
                target = parse_qs(parsed.query).get("uddg", [])
                if target:
                    href = target[0]
            if title and href.startswith(("https://", "http://")):
                results.append(f"**[{title}]({href})**\n{snippet}")
        if results:
            return "Web search results (open the source links to verify details):\n\n" + "\n\n".join(results)
        return _ddg_instant(q, max_results)
    except Exception as e:
        logger.warning("web search failed: %s", e)
        return _ddg_instant(q, max_results)


def _ddg_instant(q: str, max_results: int = 5) -> str:
    try:
        r = HTTP.get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
        if r.status_code != 200:
            return f"Search currently unavailable ({r.status_code})."
        data = r.json()
        parts: list = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:900])
        for topic in (data.get("RelatedTopics") or [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"][:350])
            elif isinstance(topic, dict) and topic.get("Topics"):
                for t in topic["Topics"][:2]:
                    if t.get("Text"):
                        parts.append(t["Text"][:350])
        return "\n".join(parts)[:2800] if parts else "No usable search results were returned. Try a more specific query."
    except Exception as e:
        return f"Search failed: {type(e).__name__}"


def run_code_sandboxed(language: str, code: str) -> str:
    lang = (language or "python").lower().strip()
    code = (code or "")[: settings.max_code_len]
    if not code.strip():
        return "empty code"
    lang_map = {"py": "python", "js": "javascript", "ts": "typescript", "sh": "bash"}
    lang = lang_map.get(lang, lang)
    try:
        r = HTTP.post(
            f"{settings.piston_url}/execute",
            json={"language": lang, "version": "*", "files": [{"content": code}]},
            timeout=20,
        )
        if r.status_code != 200:
            return f"sandbox error ({r.status_code}): {r.text[:300]}"
        data = r.json()
        run = data.get("run") or {}
        out = (run.get("stdout") or "") + (run.get("stderr") or "")
        return (out or "(no output)")[:4000]
    except Exception as e:
        return f"sandbox error: {e}"


def safe_calc(expression: str) -> dict:
    expr = (expression or "").strip()
    if not expr or len(expr) > 200:
        return {"ok": False, "error": "invalid expression"}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval(tree)
        if not isinstance(value, (int, float)):
            return {"ok": False, "error": "non-numeric result"}
        return {"ok": True, "value": value}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _pdf_sanitize(text: str) -> str:
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def _pdf_escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _simple_pdf(lines: list, title: str = "") -> bytes:
    content_lines = ["BT", "/F1 16 Tf", "50 800 Td"]
    if title:
        content_lines.append(f"({_pdf_escape(title[:80])}) Tj")
        content_lines.append("0 -24 Td")
        content_lines.append("/F1 11 Tf")
    else:
        content_lines.append("/F1 11 Tf")
    for line in lines:
        line = (line or "")[:110]
        content_lines.append(f"({_pdf_escape(line)}) Tj")
        content_lines.append("0 -14 Td")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", "replace")
    objects = []
    objects.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objects.append(b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
    objects.append(
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n"
    )
    objects.append(
        f"4 0 obj<< /Length {len(stream)} >>stream\n".encode() + stream + b"\nendstream\nendobj\n"
    )
    objects.append(b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode())
    return bytes(out)


def _require_fpdf():
    try:
        from fpdf import FPDF
        return FPDF
    except ImportError:
        return None


def generate_pdf_document(title: str, sections: list | None = None) -> bytes:
    FPDF = _require_fpdf()
    if FPDF is None:
        lines = []
        for sec in (sections or [])[:25]:
            if isinstance(sec, dict):
                if sec.get("heading"):
                    lines.append(str(sec["heading"]))
                for line in str(sec.get("content") or "").split("\n"):
                    lines.append(line)
        return _simple_pdf(lines, title=title or "Document")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _pdf_sanitize(title or "Document"))
    pdf.ln(3)
    for sec in (sections or [])[:25]:
        if not isinstance(sec, dict):
            continue
        if sec.get("heading"):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, _pdf_sanitize(str(sec["heading"])))
        pdf.set_font("Helvetica", "", 11)
        for line in str(sec.get("content") or "").split("\n"):
            pdf.multi_cell(0, 6, _pdf_sanitize(line))
        pdf.ln(2)
    return bytes(pdf.output())


def generate_cv_pdf(cv: dict) -> bytes:
    FPDF = _require_fpdf()
    cv = cv or {}
    if FPDF is None:
        lines = [str(cv.get("name") or "Name"), str(cv.get("title") or "")]
        if cv.get("summary"):
            lines += ["", "SUMMARY", str(cv["summary"])]
        if cv.get("experience"):
            lines += ["", "EXPERIENCE"]
            for job in (cv.get("experience") or [])[:12]:
                if isinstance(job, dict):
                    lines.append(f"{job.get('role','')} - {job.get('company','')}")
                    for b in (job.get("bullets") or [])[:6]:
                        lines.append(f"  - {b}")
        if cv.get("skills"):
            sk = cv["skills"]
            lines += ["", "SKILLS", ", ".join(sk) if isinstance(sk, list) else str(sk)]
        return _simple_pdf(lines, title="")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _pdf_sanitize(str(cv.get("name") or "Name")), ln=True)
    if cv.get("title"):
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, _pdf_sanitize(str(cv["title"])), ln=True)
    contact_bits = [str(cv.get(k)) for k in ("email", "phone", "location", "linkedin", "github") if cv.get(k)]
    if contact_bits:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _pdf_sanitize(" | ".join(contact_bits)), ln=True)
    pdf.ln(3)
    if cv.get("summary"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "SUMMARY", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _pdf_sanitize(str(cv["summary"])))
        pdf.ln(2)
    if cv.get("experience"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "EXPERIENCE", ln=True)
        for job in (cv.get("experience") or [])[:12]:
            if not isinstance(job, dict):
                continue
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, _pdf_sanitize(f"{job.get('role', '')} — {job.get('company', '')}"), ln=True)
            if job.get("dates"):
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, _pdf_sanitize(str(job["dates"])), ln=True)
            pdf.set_font("Helvetica", "", 11)
            for b in (job.get("bullets") or [])[:8]:
                pdf.multi_cell(0, 6, _pdf_sanitize(f"- {b}"))
            pdf.ln(1)
    if cv.get("education"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "EDUCATION", ln=True)
        for ed in (cv.get("education") or [])[:6]:
            if not isinstance(ed, dict):
                continue
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, _pdf_sanitize(f"{ed.get('degree', '')} — {ed.get('school', '')}"), ln=True)
            if ed.get("dates"):
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, _pdf_sanitize(str(ed["dates"])), ln=True)
            pdf.ln(1)
    if cv.get("skills"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "SKILLS", ln=True)
        pdf.set_font("Helvetica", "", 11)
        skills = cv["skills"]
        if isinstance(skills, list):
            skills = ", ".join(str(s) for s in skills)
        pdf.multi_cell(0, 6, _pdf_sanitize(str(skills))
    return bytes(pdf.output())
