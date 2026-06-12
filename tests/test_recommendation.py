# tests/test_recommendation.py

from app.agents import RecommendationAgent

agent = RecommendationAgent()

results = agent.run(
    "best earbuds under 1500"
)

print(results)