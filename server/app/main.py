"""JagX v12 — server entrypoint.

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from .api import create_app
from .config import APP_NAME, VERSION, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("jagx")

app = create_app()


@app.on_event("startup")
def _startup() -> None:
    logger.info("%s v%s starting", APP_NAME, VERSION)
    if getattr(settings, "jwt_is_default", False):
        logger.warning(
            "JAGX_JWT_SECRET is not set — using an insecure development default. "
            "Set it before exposing this server."
        )
    providers = settings.provider_status()
    if providers:
        logger.info("LLM chain: %s", ", ".join(f"{p['name']}({p['model']})" for p in providers))
    else:
        logger.warning(
            "No LLM provider configured. JagX is local-first: start Ollama "
            "(`ollama serve` + `ollama pull qwen2.5:7b`) — no API key required."
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=False)
