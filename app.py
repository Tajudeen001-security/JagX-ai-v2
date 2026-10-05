"""Compatibility shim — prefer python main.py or uvicorn jagx_api.app:app."""
from jagx_api.app import app
__all__ = ["app"]
