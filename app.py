"""JagX AI 7.4.0 — chat, tools, MCP, jobs, vault, code, memory, calc. Created by JagX & JRILICENSE."""
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

def _bind(mod):
    mod.app = globals().get("app")
    if globals().get("HTTP") is not None:
        mod.HTTP = globals()["HTTP"]

try:
    import jagx_extensions as _jx
    _bind(_jx)
    _jx.TRAINING_DATA_FILE = globals().get("TRAINING_DATA_FILE", "jagx_training_data.jsonl")
    if "LOCAL_KB" in globals():
        _jx.LOCAL_KB = globals()["LOCAL_KB"]
    if "load_local_brain" in globals():
        _jx.load_local_brain = globals()["load_local_brain"]
    _jx.register_extension_routes()
except Exception as e:
    _log.warning("extensions: %s", e)

try:
    import jagx_files as _jf
    _bind(_jf)
    _jf.HF_TOKEN = os.environ.get("HF_TOKEN", "")
    _jf.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    _jf.register_file_routes()
except Exception as e:
    _log.warning("files: %s", e)

try:
    import jagx_mcp as _jm
    _bind(_jm)
    _jm.register_mcp_routes(globals().get("app"))
except Exception as e:
    _log.warning("mcp: %s", e)

try:
    import jagx_jobs as _jj
    _bind(_jj)
    if "generate_response" in globals():
        _jj.generate_response = globals()["generate_response"]
    _jj.register_job_routes(globals().get("app"))
except Exception as e:
    _log.warning("jobs: %s", e)

try:
    import jagx_vault as _jv
    _bind(_jv)
    _jv.register_vault_routes(globals().get("app"))
except Exception as e:
    _log.warning("vault: %s", e)

try:
    import jagx_codebot as _jc
    _bind(_jc)
    _jc.register_code_routes(globals().get("app"))
except Exception as e:
    _log.warning("codebot: %s", e)

try:
    import jagx_memory as _jmem
    _bind(_jmem)
    _jmem.register_memory_routes(globals().get("app"))
except Exception as e:
    _log.warning("memory: %s", e)

try:
    import jagx_sandbox as _js
    _bind(_js)
    _js.register_sandbox_routes(globals().get("app"))
except Exception as e:
    _log.warning("sandbox: %s", e)

_log.info("JagX 7.4 modules loaded")
