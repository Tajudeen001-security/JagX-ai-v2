"""JagX AI 7.0.0 — core + free news/maps/self-learn. Created by JagX & JRILICENSE."""
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

# Free extensions: live news (RSS), OpenStreetMap, weather, idle self-learn
try:
    import jagx_extensions as _jx

    # Inject core objects into extension module
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
