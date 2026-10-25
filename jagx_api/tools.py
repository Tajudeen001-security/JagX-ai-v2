"""Built-in tools: web search, code sandbox, calc, PDF, CV, portfolio, fetch_url."""
from __future__ import annotations
import base64
from pathlib import Path
_dir = Path(__file__).resolve().parent
_SRC = base64.b64decode(
    (_dir / "_tools_b64_a.txt").read_text() + (_dir / "_tools_b64_b.txt").read_text()
).decode("utf-8")
exec(compile(_SRC, str(_dir / "_tools_impl.py"), "exec"), globals())
