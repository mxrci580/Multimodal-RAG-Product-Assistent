from app.agents import ComparisonAgent

agent = ComparisonAgent()

results = agent.run(
    "compare boat and noise earbuds"
)

print(results)