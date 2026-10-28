from app.vision import VisionAnalyzer
from app.retrieval import ProductRetriever
from app.generator import ProductGenerator


class MultimodalAgent:

    def __init__(self, api_key=None):

        self.vision = VisionAnalyzer(api_key=api_key)
        self.retriever = ProductRetriever(api_key=api_key)
        self.generator = ProductGenerator(api_key=api_key)

    def run(self, image_path):

        description = self.vision.analyze_image(
            image_path
        )

        results = self.retriever.search(
            query=description,
            top_k=5
        )

        docs = results["documents"][0]

        query = f"""
Image Description:
{description}

Recommend the closest matching product from the retrieved products.

Explain why it matches the uploaded image.
"""

        answer = self.generator.generate_answer(
            query,
            docs
        )

        return {
            "answer": answer,
            "products": results["metadatas"][0]
        }