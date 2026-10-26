import re
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


def clean_image_url(url: str) -> str:
    """Strip expired Amazon CDN experiment path segments (e.g. /images/W/WEBP_.../images/)."""
    if not url:
        return ""
    return re.sub(r'/images/W/[^/]+/images/', '/images/', str(url).strip())


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

            # Parse structured numeric and category fields
            clean_price = re.sub(r"[^\d.]", "", str(row["discounted_price"]))
            try:
                num_price = float(clean_price) if clean_price else 0.0
            except ValueError:
                num_price = 0.0

            try:
                num_rating = float(str(row["rating"]))
            except ValueError:
                num_rating = 0.0

            cat_parts = [p.strip() for p in str(row["category"]).split("|") if p.strip()]
            main_cat = cat_parts[0] if cat_parts else "Other"
            sub_cat = cat_parts[-1] if cat_parts else "Other"
            p_name = str(row["product_name"])
            brand = p_name.split()[0] if p_name.split() else "Unknown"

            self.collection.add(
                ids=[product_id],
                documents=[product_text],
                embeddings=[embedding],
                metadatas=[{
                    "product_id": product_id,
                    "product_name": p_name,
                    "category": str(row["category"]),
                    "main_category": main_cat,
                    "subcategory": sub_cat,
                    "brand": brand,
                    "price": str(row["discounted_price"]),
                    "numeric_price": num_price,
                    "rating": str(row["rating"]),
                    "numeric_rating": num_rating,
                    "image_url": clean_image_url(row["img_link"]),
                    "product_link": str(row["product_link"])
                }]
            )

        print("\nIngestion Complete")


if __name__ == "__main__":

    ingestion = ProductIngestion()

    ingestion.ingest_csv(
        "data/amazon.csv"
    )