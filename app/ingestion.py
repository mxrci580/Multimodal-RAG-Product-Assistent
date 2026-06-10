import pandas as pd
from tqdm import tqdm

import chromadb
from google import genai

from app.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    COLLECTION_NAME
)


class ProductIngestion:

    def __init__(self):

        self.client = genai.Client(
            api_key=GOOGLE_API_KEY
        )

        self.chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def generate_embedding(self, text):

        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

        return response.embeddings[0].values

    def ingest_csv(self, csv_path):

        df = pd.read_csv(csv_path)

        print(f"\nFound {len(df)} products\n")

        for idx, row in tqdm(df.iterrows(), total=len(df)):

            product_text = f"""
            Product Name: {row['name']}
            Category: {row['category']}
            Description: {row['description']}
            Price: {row['price']}
            """

            embedding = self.generate_embedding(
                product_text
            )

            self.collection.add(
                ids=[str(idx)],
                documents=[product_text],
                embeddings=[embedding],
                metadatas=[{
                    "name": row["name"],
                    "category": row["category"],
                    "price": float(row["price"])
                }]
            )

        print("\nIngestion Complete")


if __name__ == "__main__":

    ingestion = ProductIngestion()

    ingestion.ingest_csv(
        "data/products.csv"
    )