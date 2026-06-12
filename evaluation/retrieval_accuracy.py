from app.retrieval import ProductRetriever
from evaluation.sample_questions import evaluation_data

retriever = ProductRetriever()

hits = 0

for item in evaluation_data:

    results = retriever.search(
        item["question"],
        top_k=3
    )

    retrieved = [
        m["product_name"].lower()
        for m in results["metadatas"][0]
    ]

    gt = item["ground_truth"].lower()

    if any(gt in product for product in retrieved):
        hits += 1

accuracy = hits / len(evaluation_data)

print(f"Top-3 Accuracy: {accuracy:.2%}")