from app.rag_chain import ProductRAG

rag = ProductRAG()

query = input("Ask: ")

answer = rag.generate_answer(
    query
)

print("\nANSWER\n")
print(answer)