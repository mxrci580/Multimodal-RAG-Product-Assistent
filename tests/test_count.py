import chromadb

client = chromadb.PersistentClient(
    path="./vector_db"
)

collection = client.get_collection(
    "products"
)

print("Number of products:", collection.count())