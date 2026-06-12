from app.retrieval import ProductRetriever
from app.generator import ProductGenerator

retriever = ProductRetriever()
generator = ProductGenerator()

query = "best wireless earbuds"

results = retriever.search(
    query=query,
    top_k=3
)

documents = results["documents"][0]

answer = generator.generate_answer(
    query,
    documents
)

print(answer)