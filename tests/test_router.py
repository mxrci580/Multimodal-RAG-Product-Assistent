from app.router import QueryRouter

router = QueryRouter()

queries = [
    "Compare iPhone 15 and Samsung S24",
    "Recommend headphones under 3000",
    "Tell me about Apple Watch"
]

for query in queries:

    route = router.route(query)

    print(
        f"Query: {query}"
    )

    print(
        f"Agent: {route}"
    )

    print("-" * 50)