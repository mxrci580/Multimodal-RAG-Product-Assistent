import os
from dotenv import load_dotenv

load_dotenv()

# API Keys: Support both GEMINI_API_KEY and GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = GOOGLE_API_KEY

# Model Configurations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Vector Database Path & Collection
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")

# Deployment Network Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
