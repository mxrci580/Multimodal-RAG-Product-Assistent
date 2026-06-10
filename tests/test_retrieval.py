from app.retrieval import ProductRetriever

retriever = ProductRetriever()

query = input("Ask: ")

results = retriever.search(query)

print("\nRESULTS\n")

for doc in results["documents"][0]:

    print(doc)
    print("-" * 50)