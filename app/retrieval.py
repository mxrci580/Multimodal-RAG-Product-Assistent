import chromadb
from google import genai

from app.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    COLLECTION_NAME
)


class ProductRetriever:

    def __init__(self):

        self.client = genai.Client(
            api_key=GOOGLE_API_KEY
        )

        self.chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH
        )

        self.collection = self.chroma_client.get_collection(
            name=COLLECTION_NAME
        )

    def generate_embedding(self, text):

        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
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