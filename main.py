"""
JagX Backend entrypoint for Render.
Dockerfile runs: python main.py
Loads the full API from app.py (v7+).
Created by JagX & JRILICENSE
"""
import os

# Prefer the full feature app (tools, PDF, image, stronger prompts)
try:
    from app import app  # noqa: F401
except Exception as e:
    raise SystemExit(f"Failed to import app.py: {e}") from e

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
