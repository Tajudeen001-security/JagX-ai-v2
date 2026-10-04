"""JagX Backend entrypoint. Dockerfile: python main.py"""
import os
from app import app
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
