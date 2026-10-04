"""JagX AI 7.1.0 — core + news/maps + files + MCP connectors. Created by JagX & JRILICENSE."""
import os
import urllib.request
import logging

_log = logging.getLogger("jagx-ai")

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

# Free extensions: live news, OpenStreetMap, weather, idle self-learn
try:
    import jagx_extensions as _jx

    _jx.app = globals().get("app")
    if globals().get("HTTP") is not None:
        _jx.HTTP = globals()["HTTP"]
    _jx.TRAINING_DATA_FILE = globals().get("TRAINING_DATA_FILE", "jagx_training_data.jsonl")
    if "LOCAL_KB" in globals():
        _jx.LOCAL_KB = globals()["LOCAL_KB"]
    if "load_local_brain" in globals():
        _jx.load_local_brain = globals()["load_local_brain"]

    _jx.register_extension_routes()
    _log.info("JagX extensions active: /news /geo /weather /self_learn /extensions")
except Exception as _ext_err:
    _log.warning("jagx_extensions not loaded: %s", _ext_err)

# File readers: screenshots/images, PDF, ZIP
try:
    import jagx_files as _jf

    _jf.app = globals().get("app")
    if globals().get("HTTP") is not None:
        _jf.HTTP = globals()["HTTP"]
    _jf.HF_TOKEN = os.environ.get("HF_TOKEN", "")
    _jf.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    _jf.register_file_routes()
    _log.info("JagX file readers active: /read_file /read_image /read_pdf /read_zip")
except Exception as _file_err:
    _log.warning("jagx_files not loaded: %s", _file_err)

# MCP connectors proxy (remote MCP + built-in free tools)
try:
    import jagx_mcp as _jm

    _jm.app = globals().get("app")
    if globals().get("HTTP") is not None:
        _jm.HTTP = globals()["HTTP"]
    _jm.register_mcp_routes(globals().get("app"))
    _log.info("JagX MCP active: /mcp/info /mcp/run /mcp/list_tools /mcp/call")
except Exception as _mcp_err:
    _log.warning("jagx_mcp not loaded: %s", _mcp_err)
