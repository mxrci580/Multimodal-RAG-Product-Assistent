from app.multimodal_agent import MultimodalAgent

agent = MultimodalAgent()

response = agent.run(
    "data/test_images/image.png"
)

print(response)