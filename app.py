"""JagX AI 7.0.0 — loads full backend. Created by JagX & JRILICENSE."""
import os
import urllib.request

# Prefer local full source if present (src_part*.txt), else fetch public complete main
_src = None
try:
    from pathlib import Path
    _base = Path(__file__).parent
    _parts = sorted(_base.glob("src_part*.txt"), key=lambda p: int(p.stem.replace("src_part", "")))
    if _parts and sum(p.stat().st_size for p in _parts) > 20000:
        _src = "".join(p.read_text() for p in _parts)
except Exception:
    _src = None

if not _src:
    url = os.environ.get(
        "JAGX_FULL_SOURCE_URL",
        "https://raw.githubusercontent.com/wantajudeen/jagx-backend/main/main.py",
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        _src = r.read().decode("utf-8")

exec(_src, globals())
