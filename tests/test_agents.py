from app.agents import SearchAgent

agent = SearchAgent()

results = agent.run(
    "wireless earbuds"
)

print(results)