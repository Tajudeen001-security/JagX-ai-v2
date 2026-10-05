"""JagX Backend entrypoint. Dockerfile: python main.py"""

from __future__ import annotations

import os

import uvicorn

from jagx_api.app import app
from jagx_api.config import settings

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", str(settings.port))),
        log_level="info",
    )
