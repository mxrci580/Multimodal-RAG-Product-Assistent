from app.retrieval import ProductRetriever
from evaluation.sample_questions import evaluation_data

retriever = ProductRetriever()

for item in evaluation_data:

    results = retriever.search(
        item["question"],
        top_k=3
    )

    print("\nQUESTION:")
    print(item["question"])

    print("\nEXPECTED:")
    print(item["ground_truth"])

    print("\nRETRIEVED:")

    for metadata in results["metadatas"][0]:
        print(
            metadata["product_name"]
        )

    print("-" * 50)