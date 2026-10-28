from google import genai
from app.config import GOOGLE_API_KEY, LLM_MODEL, FALLBACK_LLM_MODELS
from app.gemini_utils import generate_content_with_retry


class ProductGenerator:

    def __init__(self, api_key=None):
        key = api_key or GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "Gemini API Key is missing. Please configure GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        self.client = genai.Client(
            api_key=key
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

        response = generate_content_with_retry(
            client=self.client,
            model=LLM_MODEL,
            contents=prompt,
            fallback_models=FALLBACK_LLM_MODELS,
            max_retries=3,
            initial_delay=2.0
        )

        return response.text