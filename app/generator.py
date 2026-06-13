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
You are an AI product recommendation assistant.

Use the retrieved products to answer the user's query.

Recommend the closest matching products and explain why they are relevant.

Do NOT invent products that are not present in the retrieved results.

If some details from the query are missing in the retrieved products,
focus on the available features and recommend the best matches.

User Query:
{query}

Retrieved Products:
{context}

Provide:
1. Best matching product
2. Why it matches the query
3. Alternative options if available
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text