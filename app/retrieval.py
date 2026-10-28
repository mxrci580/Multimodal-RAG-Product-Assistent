import chromadb
from google import genai

from app.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    COLLECTION_NAME
)
from app.gemini_utils import embed_content_with_retry


class ProductRetriever:

    def __init__(self, api_key=None):
        key = api_key or GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "Gemini API Key is missing. Please configure GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        self.client = genai.Client(
            api_key=key
        )

        self.chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH
        )

        self.collection = self.chroma_client.get_collection(
            name=COLLECTION_NAME
        )

    def generate_embedding(self, text):

        response = embed_content_with_retry(
            client=self.client,
            model=EMBEDDING_MODEL,
            contents=text,
            max_retries=3,
            initial_delay=2.0
        )

        return response.embeddings[0].values


    def search(self, query, top_k=3):

        query_embedding = self.generate_embedding(
            query
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results