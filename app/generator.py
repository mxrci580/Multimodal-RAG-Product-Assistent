from google import genai
from app.config import GOOGLE_API_KEY

class ProductGenerator:

    def __init__(self):

        self.client = genai.Client(
            api_key=GOOGLE_API_KEY
        )

    def generate_answer(
        self,
        query,
        retrieved_products
    ):

        context = "\n\n".join(
            retrieved_products
        )

        prompt = f"""
You are a product recommendation assistant.

Use ONLY the retrieved products below.

Do NOT invent products, features, prices, ratings, or specifications.

If the answer cannot be determined from the retrieved products,
say that the information is unavailable.

User Query:
{query}

Retrieved Products:
{context}

Provide a helpful recommendation with reasoning.
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text