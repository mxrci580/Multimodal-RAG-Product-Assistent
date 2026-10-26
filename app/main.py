import os
import tempfile
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.logger import logger
from app.config import HOST, PORT, GOOGLE_API_KEY
from app.rag_chain import ProductRAG
from app.multimodal_agent import MultimodalAgent

# Initialize FastAPI application
app = FastAPI(
    title="Multimodal RAG Product Assistant",
    description="Production-ready FastAPI service for multimodal e-commerce product search, retrieval, and recommendation.",
    version="1.0.0",
)

# Enable CORS for frontend clients (Streamlit, React, mobile apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons for RAG pipelines
_rag_instance: Optional[ProductRAG] = None
_multimodal_instance: Optional[MultimodalAgent] = None


def get_rag() -> ProductRAG:
    """Lazy loader for ProductRAG to handle cold starts cleanly."""
    global _rag_instance
    if _rag_instance is None:
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable is not configured.",
            )
        try:
            _rag_instance = ProductRAG()
        except Exception as e:
            logger.error(f"Failed to initialize ProductRAG: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize retrieval engine: {str(e)}",
            )
    return _rag_instance


def get_multimodal_agent() -> MultimodalAgent:
    """Lazy loader for MultimodalAgent."""
    global _multimodal_instance
    if _multimodal_instance is None:
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable is not configured.",
            )
        try:
            _multimodal_instance = MultimodalAgent()
        except Exception as e:
            logger.error(f"Failed to initialize MultimodalAgent: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize multimodal agent: {str(e)}",
            )
    return _multimodal_instance


# Pydantic Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="User shopping search or recommendation query", example="best wireless mouse under 1000")


class QueryResponse(BaseModel):
    query: str
    answer: str


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str


@app.get("/", tags=["General"])
def home() -> Dict[str, Any]:
    """Root endpoint providing service status and available documentation."""
    return {
        "service": "Multimodal RAG Product Assistant",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Health check probe endpoint for Render / Kubernetes."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="multimodal-rag-assistant",
    )


@app.post("/ask", response_model=QueryResponse, tags=["Retrieval & Recommendation"])
def ask_question(request: QueryRequest) -> QueryResponse:
    """
    Core RAG endpoint:
    Retrieves candidate products from ChromaDB and generates an evidence-grounded answer via Gemini.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    try:
        rag = get_rag()
        answer = rag.generate_answer(request.query)
        return QueryResponse(query=request.query, answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendation: {str(e)}",
        )


@app.post("/ask-multimodal", tags=["Retrieval & Recommendation"])
async def ask_multimodal(
    file: UploadFile = File(..., description="Product image file (JPEG, PNG, WEBP)"),
    additional_query: Optional[str] = Form(None, description="Optional extra user query/constraint"),
) -> Dict[str, Any]:
    """
    Multimodal RAG endpoint:
    Accepts an uploaded product image, extracts visual features via Gemini Vision,
    searches catalog ChromaDB, and generates a visual-grounded recommendation.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image (JPEG, PNG, WEBP).",
        )

    temp_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        agent = get_multimodal_agent()
        result = agent.run(temp_path)

        return {
            "status": "success",
            "filename": file.filename,
            "answer": result.get("answer"),
            "products": result.get("products", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing multimodal request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multimodal recommendation error: {str(e)}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
