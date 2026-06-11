class QueryRouter:

    def route(self, query):

        query = query.lower()

        comparison_keywords = [
            "compare",
            "vs",
            "versus",
            "difference"
        ]

        recommendation_keywords = [
            "recommend",
            "suggest",
            "best",
            "top"
        ]

        for word in comparison_keywords:
            if word in query:
                return "comparison"

        for word in recommendation_keywords:
            if word in query:
                return "recommendation"

        return "search"