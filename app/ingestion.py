import pandas as pd
from tqdm import tqdm
import time

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

        existing_ids = set(
            self.collection.get()["ids"]
        )

        print(
            f"Found {len(existing_ids)} existing products"
        )

        print(f"\nFound {len(df)} products\n")

        for idx, row in tqdm(df.iterrows(), total=len(df)):

            product_id = str(row["product_id"])

            if product_id in existing_ids:
                continue

            product_text = f"""
Product Name: {row['product_name']}
Category: {row['category']}
Description: {row['about_product']}
Price: {row['discounted_price']}
Rating: {row['rating']}
"""

            embedding = None

            for attempt in range(3):

                try:

                    embedding = self.generate_embedding(
                        product_text
                    )

                    break

                except Exception as e:

                    print(
                        f"Attempt {attempt + 1} failed: {e}"
                    )

                    time.sleep(10)

            if embedding is None:

                print(
                    f"Skipping product {product_id}"
                )

                continue

            self.collection.add(
                ids=[product_id],
                documents=[product_text],
                embeddings=[embedding],
                metadatas=[{
                    "product_id": product_id,
                    "product_name": str(row["product_name"]),
                    "category": str(row["category"]),
                    "price": str(row["discounted_price"]),
                    "rating": str(row["rating"]),
                    "image_url": str(row["img_link"]),
                    "product_link": str(row["product_link"])
                }]
            )

        print("\nIngestion Complete")


if __name__ == "__main__":

    ingestion = ProductIngestion()

    ingestion.ingest_csv(
        "data/amazon.csv"
    )