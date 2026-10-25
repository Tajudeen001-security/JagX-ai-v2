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
from urllib.parse import quote, urlparse, parse_qs

from jagx_api.config import settings
from jagx_api.providers import HTTP

logger = logging.getLogger("jagx-ai")

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod,
}


def free_web_search(query: str, max_results: int = 6) -> str:
    q = (query or "").strip()[:220]
    if not q:
        return "empty query"
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(q)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; JagXAI/9.7)"}
        r = HTTP.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return _ddg_instant(q, max_results)
        results = []
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td)>', re.DOTALL,
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
            return "Web search results:\n\n" + "\n\n".join(results)
        return _ddg_instant(q, max_results)
    except Exception as e:
        logger.warning("web search failed: %s", e)
        return _ddg_instant(q, max_results)


def _ddg_instant(q: str, max_results: int = 5) -> str:
    try:
        r = HTTP.get("https://api.duckduckgo.com/", params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}, timeout=10)
        if r.status_code != 200:
            return f"Search currently unavailable ({r.status_code})."
        data = r.json()
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:900])
        for topic in (data.get("RelatedTopics") or [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"][:350])
            elif isinstance(topic, dict) and topic.get("Topics"):
                for t in topic["Topics"][:2]:
                    if t.get("Text"):
                        parts.append(t["Text"][:350])
        return "\n".join(parts)[:2800] if parts else "No usable search results."
    except Exception as e:
        return f"Search failed: {type(e).__name__}"


def run_code_sandboxed(language: str, code: str) -> str:
    lang = (language or "python").lower().strip()
    code = (code or "")[: settings.max_code_len]
    if not code.strip():
        return "empty code"
    lang = {"py": "python", "js": "javascript", "ts": "typescript", "sh": "bash"}.get(lang, lang)
    try:
        r = HTTP.post(f"{settings.piston_url}/execute", json={"language": lang, "version": "*", "files": [{"content": code}]}, timeout=20)
        if r.status_code != 200:
            return f"sandbox error ({r.status_code}): {r.text[:300]}"
        run = (r.json().get("run") or {})
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
        value = _eval(ast.parse(expr, mode="eval"))
        if not isinstance(value, (int, float)):
            return {"ok": False, "error": "non-numeric result"}
        return {"ok": True, "value": value}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
        content_lines.append(f"({_pdf_escape((line or '')[:110])}) Tj")
        content_lines.append("0 -14 Td")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", "replace")
    objects = [
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n",
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n",
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n",
        f"4 0 obj<< /Length {len(stream)} >>stream\n".encode() + stream + b"\nendstream\nendobj\n",
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n",
    ]
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


def generate_pdf_document(title: str, sections: list | None = None) -> bytes:
    lines = []
    for sec in (sections or [])[:25]:
        if isinstance(sec, dict):
            if sec.get("heading"):
                lines.append(str(sec["heading"]))
            for line in str(sec.get("content") or "").split("\n"):
                lines.append(line)
    return _simple_pdf(lines, title=title or "Document")


def generate_cv_pdf(cv: dict) -> bytes:
    cv = cv or {}
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


def generate_portfolio_html(data: dict) -> str:
    data = data or {}
    name = escape(str(data.get("name") or "My Portfolio"))
    title = escape(str(data.get("title") or ""))
    about = escape(str(data.get("about") or ""))
    projects_html = ""
    for p in (data.get("projects") or [])[:20]:
        if not isinstance(p, dict):
            continue
        pname = escape(str(p.get("name") or "Project"))
        pdesc = escape(str(p.get("description") or ""))
        projects_html += f'<div class="card"><h3>{pname}</h3><p>{pdesc}</p></div>'
    skills = data.get("skills") or []
    skills_str = escape(", ".join(str(s) for s in skills) if isinstance(skills, list) else str(skills))
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{name}</title></head>
<body><h1>{name}</h1><p>{title}</p><h2>About</h2><p>{about}</p>
<h2>Projects</h2>{projects_html or "<p>None</p>"}
<h2>Skills</h2><p>{skills_str or "N/A"}</p>
<footer>Generated by JagX AI</footer></body></html>"""


def image_url_from_prompt(prompt: str, width: int = 1024, height: int = 1024) -> str:
    p = quote((prompt or "abstract art").strip()[:300])
    return f"https://image.pollinations.ai/prompt/{p}?width={int(width)}&height={int(height)}&nologo=true"


def attachment_pdf(filename: str, pdf_bytes: bytes) -> dict:
    return {"type": "pdf", "filename": filename, "content_base64": base64.b64encode(pdf_bytes).decode("ascii"), "size_bytes": len(pdf_bytes)}


def attachment_html(filename: str, html: str) -> dict:
    raw = (html or "").encode("utf-8")
    return {"type": "html", "filename": filename, "content_base64": base64.b64encode(raw).decode("ascii"), "size_bytes": len(raw)}


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
            return "PDF generated successfully and attached.", attachment_pdf("document.pdf", generate_pdf_document(str(inp.get("title") or "Document"), inp.get("sections") if isinstance(inp.get("sections"), list) else []))
        except Exception as e:
            return f"PDF generation failed: {e}", None
    if name == "generate_cv":
        try:
            return "CV generated successfully and attached.", attachment_pdf("cv.pdf", generate_cv_pdf(inp))
        except Exception as e:
            return f"CV generation failed: {e}", None
    if name == "generate_portfolio":
        try:
            return "Portfolio generated successfully and attached.", attachment_html("portfolio.html", generate_portfolio_html(inp))
        except Exception as e:
            return f"Portfolio generation failed: {e}", None
    if name == "generate_image":
        prompt = str(inp.get("prompt") or "abstract art")
        return f"Image generated for: {prompt}", {"type": "image", "url": image_url_from_prompt(prompt)}
    if name == "fetch_url":
        from jagx_api.browser_tool import fetch_url as _fetch
        return _fetch(str(inp.get("url") or "")), None
    return f"unknown tool: {name}", None
