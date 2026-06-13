from app.vision import VisionAnalyzer
from app.retrieval import ProductRetriever

vision = VisionAnalyzer()
retriever = ProductRetriever()

description = vision.analyze_image(
    "data/test_images/image.png"
)
print(len(description))
print(description[:500])

results = retriever.search(
    query=description,
    top_k=5
)

for metadata in results["metadatas"][0]:

    print("\n")
    print(metadata["product_name"])
    print(metadata["price"])
    print(metadata["rating"])