from app.router import QueryRouter
from app.agents import (
    SearchAgent,
    ComparisonAgent,
    RecommendationAgent
)

router = QueryRouter()

query = input("Enter Query: ")

route = router.route(query)

if route == "comparison":
    agent = ComparisonAgent()

elif route == "recommendation":
    agent = RecommendationAgent()

else:
    agent = SearchAgent()

print(agent.run(query))