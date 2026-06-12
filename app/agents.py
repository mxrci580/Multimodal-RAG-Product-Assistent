from app.retrieval import ProductRetriever
import re

class SearchAgent:

    def __init__(self):
        self.retriever = ProductRetriever()

    def run(self, query):
        results = self.retriever.search(
            query=query,
            top_k=3
        )

        return results


class ComparisonAgent:

    def __init__(self):

        self.retriever = ProductRetriever()

    def run(self, query):

        results = self.retriever.search(
            query=query,
            top_k=2
        )

        comparison = []

        for metadata in results["metadatas"][0]:

            comparison.append({
                "name": metadata["product_name"],
                "price": metadata["price"],
                "rating": metadata["rating"]
            })

        return comparison

class RecommendationAgent:

    def __init__(self):

        self.retriever = ProductRetriever()

    def run(self, query):

        budget = None

        match = re.search(
            r"under\s+(\d+)",
            query.lower()
        )

        if match:
            budget = int(match.group(1))

        results = self.retriever.search(
            query=query,
            top_k=10
        )

        recommendations = []

        for metadata in results["metadatas"][0]:

            price = metadata.get(
                "price",
                "0"
            )

            price = int(
                price.replace("₹", "")
                     .replace(",", "")
            )

            if budget is None or price <= budget:

                recommendations.append({
                    "name": metadata["product_name"],
                    "price": price,
                    "rating": float(
                        metadata.get(
                            "rating",
                            0
                        )
                    )
                })

        recommendations.sort(
            key=lambda x: x["rating"],
            reverse=True
        )

        return recommendations[:5]