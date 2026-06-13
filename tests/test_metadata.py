from app.retrieval import ProductRetriever

retriever = ProductRetriever()

results = retriever.search(
    query="wireless earbuds",
    top_k=1
)

print(results["metadatas"][0][0])