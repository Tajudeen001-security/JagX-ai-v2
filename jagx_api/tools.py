"""Built-in tools: web search, code sandbox, calc, PDF, CV, portfolio."""

from __future__ import annotations

import ast
import base64
import json
import logging
import operator
from html import escape
from typing import Any, Optional
from urllib.parse import quote

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


def free_web_search(query: str, max_results: int = 5) -> str:
    q = (query or "").strip()[:200]
    if not q:
        return "empty query"
    try:
        r = HTTP.get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=12,
        )
        if r.status_code != 200:
            return f"search failed ({r.status_code})"
        data = r.json()
        parts: list = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:800])
        for topic in (data.get("RelatedTopics") or [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"][:300])
            elif isinstance(topic, dict) and topic.get("Topics"):
                for t in topic["Topics"][:2]:
                    if t.get("Text"):
                        parts.append(t["Text"][:300])
        return "\n".join(parts)[:2500] if parts else "No concise results found."
    except Exception as e:
        logger.warning("search: %s", e)
        return f"search error: {e}"


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
        return (out or "(no output)")[:3000]
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
        pdf.multi_cell(0, 6, _pdf_sanitize(str(skills)))
    return bytes(pdf.output())


def generate_portfolio_html(data: dict) -> str:
    data = data or {}
    name = escape(str(data.get("name") or "My Portfolio"))
    title = escape(str(data.get("title") or ""))
    about = escape(str(data.get("about") or ""))
    projects = data.get("projects") or []
    skills = data.get("skills") or []
    contact = data.get("contact") if isinstance(data.get("contact"), dict) else {}
    projects_html = ""
    for p in projects[:20]:
        if not isinstance(p, dict):
            continue
        pname = escape(str(p.get("name") or "Project"))
        pdesc = escape(str(p.get("description") or ""))
        plink = str(p.get("link") or "").strip()
        link_html = (
            f'<p><a href="{escape(plink)}" target="_blank" rel="noopener">View Project</a></p>'
            if plink.startswith(("http://", "https://"))
            else ""
        )
        projects_html += f'<div class="card"><h3>{pname}</h3><p>{pdesc}</p>{link_html}</div>'
    skills_str = escape(", ".join(str(s) for s in skills) if isinstance(skills, list) else str(skills))
    email = escape(str(contact.get("email") or data.get("email") or ""))
    github = escape(str(contact.get("github") or data.get("github") or ""))
    linkedin = escape(str(contact.get("linkedin") or data.get("linkedin") or ""))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name} - Portfolio</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;line-height:1.6;color:#222}}
h1{{margin-bottom:4px}}.title{{color:#555;margin-top:0}}
.card{{border:1px solid #ddd;padding:16px;margin:12px 0;border-radius:8px}}
a{{color:#0066cc}}footer{{margin-top:32px;color:#888;font-size:12px}}
</style></head><body>
<h1>{name}</h1><p class="title">{title}</p>
<h2>About</h2><p>{about}</p>
<h2>Projects</h2>{projects_html or "<p>No projects listed.</p>"}
<h2>Skills</h2><p>{skills_str or "Not specified"}</p>
<h2>Contact</h2><p>{email}<br>{github}<br>{linkedin}</p>
<footer>Generated by JagX AI</footer></body></html>"""


def image_url_from_prompt(prompt: str, width: int = 1024, height: int = 1024) -> str:
    p = quote((prompt or "abstract art").strip()[:300])
    return f"https://image.pollinations.ai/prompt/{p}?width={int(width)}&height={int(height)}&nologo=true"


def attachment_pdf(filename: str, pdf_bytes: bytes) -> dict:
    return {
        "type": "pdf",
        "filename": filename,
        "content_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        "size_bytes": len(pdf_bytes),
    }


def attachment_html(filename: str, html: str) -> dict:
    raw = (html or "").encode("utf-8")
    return {
        "type": "html",
        "filename": filename,
        "content_base64": base64.b64encode(raw).decode("ascii"),
        "size_bytes": len(raw),
    }


def run_tool(name: str, inp: dict) -> tuple:
    name = (name or "").strip()
    inp = inp if isinstance(inp, dict) else {}
    if name == "web_search":
        return free_web_search(str(inp.get("query") or "")), None
    if name == "run_code":
        return run_code_sandboxed(str(inp.get("language") or "python"), str(inp.get("code") or "")), None
    if name == "calc":
        return json.dumps(safe_calc(str(inp.get("expression") or "")), ensure_ascii=False), None
    if name == "generate_pdf":
        try:
            pdf_bytes = generate_pdf_document(
                str(inp.get("title") or "Document"),
                inp.get("sections") if isinstance(inp.get("sections"), list) else [],
            )
            return "PDF generated successfully and attached.", attachment_pdf("document.pdf", pdf_bytes)
        except Exception as e:
            return f"PDF generation failed: {e}", None
    if name == "generate_cv":
        try:
            return "CV generated successfully and attached.", attachment_pdf("cv.pdf", generate_cv_pdf(inp))
        except Exception as e:
            return f"CV generation failed: {e}", None
    if name == "generate_portfolio":
        try:
            return "Portfolio generated successfully and attached.", attachment_html(
                "portfolio.html", generate_portfolio_html(inp)
            )
        except Exception as e:
            return f"Portfolio generation failed: {e}", None
    if name == "generate_image":
        prompt = str(inp.get("prompt") or "abstract art")
        return f"Image generated for: {prompt}", {"type": "image", "url": image_url_from_prompt(prompt)}
    return f"unknown tool: {name}", None
