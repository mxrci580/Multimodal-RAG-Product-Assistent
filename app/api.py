from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from app.rag_chain import ProductRAG

app = FastAPI(
    title="Multimodal RAG Product Assistant"
)

rag = ProductRAG()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/")
def home():
    return {
        "message": "RAG Product Assistant Running"
    }


@app.post("/ask")
def ask_question(request: QueryRequest):

    try:

        answer = rag.generate_answer(
            request.query
        )

        return {
            "query": request.query,
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
     