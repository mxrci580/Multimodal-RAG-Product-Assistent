# tests/check_collection.py

import chromadb
from app.config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME
)

client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

collection = client.get_collection(
    COLLECTION_NAME
)

print(
    f"Products Stored: {collection.count()}"
)