from google import genai
import time

from app.logger import logger
from app.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    FALLBACK_LLM_MODELS
)
from app.retrieval import ProductRetriever
from app.gemini_utils import generate_content_with_retry



class ProductRAG:

    def __init__(self, api_key=None):
        key = api_key or GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "Gemini API Key is missing. Please configure GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        self.client = genai.Client(
            api_key=key
        )
        self.retriever = ProductRetriever(api_key=key)

    def generate_answer(self, query):

        start_time = time.time()

        logger.info(f"Query: {query}")

        results = self.retriever.search(
            query=query,
            top_k=3
        )

        logger.info(
            f"Retrieved {len(results['documents'][0])} documents"
        )

        context = "\n\n".join(
            results["documents"][0]
        )

        logger.debug(f"Retrieved context:\n{context}")

        prompt = f"""
You are a product assistant.

Use the context to answer the question.

Be concise.

If relevant products are found, summarize them.

Only say "I do not have enough information"
when the context is completely unrelated.

Context:
{context}

Question:
{query}
"""

        response = generate_content_with_retry(
            client=self.client,
            model=LLM_MODEL,
            contents=prompt,
            fallback_models=FALLBACK_LLM_MODELS,
            max_retries=3,
            initial_delay=2.0
        )

        response_time = round(
            time.time() - start_time,
            2
        )

        logger.info(
            f"Response Time: {response_time} sec"
        )

        return response.text