"""Built-in tools: web search, code sandbox, calc, PDF, CV, portfolio, fetch_url."""
from __future__ import annotations
from pathlib import Path
_d = Path(__file__).resolve().parent
for _name in ("_tools_p1.py", "_tools_p2.py", "_tools_p3.py"):
    _src = (_d / _name).read_text(encoding="utf-8")
    exec(compile(_src, str(_d / _name), "exec"), globals())
