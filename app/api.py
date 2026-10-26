"""
Backward-compatible API entrypoint.
Imports and exposes the FastAPI app from app.main.
"""
from app.main import app, QueryRequest, health, home, ask_question

__all__ = ["app", "QueryRequest", "health", "home", "ask_question"]