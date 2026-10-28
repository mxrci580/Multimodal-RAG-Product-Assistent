import os
from dotenv import load_dotenv

load_dotenv()

# Check Streamlit secrets if running inside Streamlit Cloud
_st_key = None
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        _st_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
except Exception:
    pass

# API Keys: Support both GEMINI_API_KEY and GOOGLE_API_KEY (from Streamlit secrets or env)
GOOGLE_API_KEY = _st_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = GOOGLE_API_KEY

# Model Configurations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
FALLBACK_LLM_MODELS = [
    m.strip()
    for m in os.getenv("FALLBACK_LLM_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash").split(",")
    if m.strip()
]

# Vector Database Path & Collection
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")

# Deployment Network Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
