from google import genai
import time

from app.logger import logger

from app.config import (
    GOOGLE_API_KEY,
    LLM_MODEL
)

from app.retrieval import ProductRetriever


class ProductRAG:

    def __init__(self):

        self.client = genai.Client(
            api_key=GOOGLE_API_KEY
        )

        self.retriever = ProductRetriever()

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

        print("\n===== CONTEXT =====\n")
        print(context)
        print("\n===================\n")

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

        response = self.client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )

        response_time = round(
            time.time() - start_time,
            2
        )

        logger.info(
            f"Response Time: {response_time} sec"
        )

        return response.text