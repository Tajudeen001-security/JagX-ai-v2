"""JagX AI 7.3.0 — core + tools + MCP + jobs + per-user code bot. Created by JagX & JRILICENSE."""
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
    _log.info("JagX extensions active")
except Exception as _ext_err:
    _log.warning("jagx_extensions not loaded: %s", _ext_err)

try:
    import jagx_files as _jf
    _jf.app = globals().get("app")
    if globals().get("HTTP") is not None:
        _jf.HTTP = globals()["HTTP"]
    _jf.HF_TOKEN = os.environ.get("HF_TOKEN", "")
    _jf.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    _jf.register_file_routes()
except Exception as _file_err:
    _log.warning("jagx_files not loaded: %s", _file_err)

try:
    import jagx_mcp as _jm
    _jm.app = globals().get("app")
    if globals().get("HTTP") is not None:
        _jm.HTTP = globals()["HTTP"]
    _jm.register_mcp_routes(globals().get("app"))
except Exception as _mcp_err:
    _log.warning("jagx_mcp not loaded: %s", _mcp_err)

try:
    import jagx_jobs as _jj
    _jj.app = globals().get("app")
    if globals().get("HTTP") is not None:
        _jj.HTTP = globals()["HTTP"]
    if "generate_response" in globals():
        _jj.generate_response = globals()["generate_response"]
    _jj.register_job_routes(globals().get("app"))
except Exception as _job_err:
    _log.warning("jagx_jobs not loaded: %s", _job_err)

try:
    import jagx_vault as _jv
    _jv.app = globals().get("app")
    _jv.register_vault_routes(globals().get("app"))
    _log.info("JagX vault active")
except Exception as _v_err:
    _log.warning("jagx_vault not loaded: %s", _v_err)

try:
    import jagx_codebot as _jc
    _jc.app = globals().get("app")
    if globals().get("HTTP") is not None:
        _jc.HTTP = globals()["HTTP"]
    _jc.register_code_routes(globals().get("app"))
    _log.info("JagX code bot active: /code/push /code/build")
except Exception as _c_err:
    _log.warning("jagx_codebot not loaded: %s", _c_err)
