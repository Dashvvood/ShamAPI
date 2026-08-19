"""Deprecated: app.qwen3 is a module (app/qwen3.py), not this directory.

Launch with:
    gunicorn app:qwen3
    uvicorn app.qwen3:app
"""
from app.qwen3 import app

__all__ = ["app"]
