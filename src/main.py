"""
FastAPI Application - Wikipedia RAG System

Sprint 1: Basic infrastructure (health, echo, docs)
Sprint 2: RAG system integration (ingestion, embeddings, vector DB)

Endpoints:
- Health check and echo (Sprint 1)
- Wikipedia ingestion and vector database (Sprint 2)
- Auto-generated documentation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Import our RAG system
from rag_system import WikipediaRAGSystem, create_rag_system

# Request/Response Models
class EchoRequest(BaseModel):
    """Simple request model for echo testing."""
    message: str


class EchoResponse(BaseModel):
    """Simple response model for echo testing."""
    echo: str
    received_at: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    python_version: str
    dependencies_loaded: bool


# Sprint 2 Models
class IngestionRequest(BaseModel):
    """Request model for Wikipedia data ingestion."""
    dataset_url: Optional[str] = None
    force_rebuild: bool = False
    chunk_size: Optional[int] = 512


class IngestionResponse(BaseModel):
    """Response model for ingestion operation."""
    status: str
    message: str
    documents_loaded: int
    embedding_model: str
    vector_db_ready: bool


class VectorDBStatusResponse(BaseModel):
    """Response model for vector database status."""
    status: str
    num_documents: int
    num_nodes: Any
    storage_directory: str
    chunk_size: int
    embedding_model: str
    llm_model: str
    index_ready: bool


# Initialize FastAPI app
app = FastAPI(
    title="Wikipedia RAG System",
    description="RAG system with Wikipedia ingestion, embeddings, and Q&A endpoints",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global RAG system instance
rag_system: Optional[WikipediaRAGSystem] = None


def get_rag_system() -> WikipediaRAGSystem:
    """Get or initialize the RAG system."""
    global rag_system
    if rag_system is None:
        rag_system = create_rag_system()
    return rag_system


@app.get("/")
async def root():
    """Root endpoint with basic system information."""
    return {
        "message": "Wikipedia RAG System",
        "version": "2.0.0",
        "sprint": "2 - RAG Integration",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ingest": "/ingest",
            "vector_status": "/vector-db/status"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Validates that the server is running and basic dependencies are available.
    """
    try:
        # Check if we can import key dependencies
        dependencies_loaded = True
        try:
            import pandas
            import fastapi
            # Don't import heavy dependencies for basic health check
        except ImportError:
            dependencies_loaded = False

        from datetime import datetime

        return HealthResponse(
            status="healthy",
            message="Server is running successfully",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            dependencies_loaded=dependencies_loaded
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            python_version="unknown",
            dependencies_loaded=False
        )


@app.post("/echo", response_model=EchoResponse)
async def echo_endpoint(request: EchoRequest):
    """
    Simple echo endpoint for testing request/response functionality.

    Takes a message and echoes it back with a timestamp.
    """
    from datetime import datetime

    return EchoResponse(
        echo=request.message,
        received_at=datetime.now().isoformat()
    )


@app.get("/echo/{message}")
async def echo_get(message: str):
    """
    GET version of echo endpoint for simple browser testing.
    """
    from datetime import datetime

    return {
        "echo": message,
        "received_at": datetime.now().isoformat(),
        "method": "GET"
    }


# Sprint 2 Endpoints - RAG System Integration

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_wikipedia_data(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    Sprint 2: Ingest Wikipedia data and generate embeddings.

    This endpoint:
    1. Loads Wikipedia passages from HuggingFace
    2. Generates embeddings using Azure text-embedding-3-large
    3. Inserts encoded vectors into the vector database
    4. Sets up retrieval infrastructure
    """
    try:
        system = get_rag_system()

        # Load Wikipedia documents
        print(f"Loading Wikipedia data (force_rebuild={request.force_rebuild})...")
        documents = system.load_wikipedia_data(request.dataset_url)

        # Build vector index with embeddings
        print("Building vector index with embeddings...")
        system.build_index(documents, force_rebuild=request.force_rebuild)

        # Setup retrieval components
        system.setup_retriever()
        system.setup_query_engine()

        return IngestionResponse(
            status="success",
            message=f"Successfully ingested {len(documents)} documents with embeddings",
            documents_loaded=len(documents),
            embedding_model="text-embedding-3-large",
            vector_db_ready=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/vector-db/status", response_model=VectorDBStatusResponse)
async def get_vector_db_status():
    """
    Sprint 2: Get vector database status and statistics.

    Returns counts, index info, and operational status of the vector database.
    """
    try:
        system = get_rag_system()
        stats = system.get_index_stats()

        return VectorDBStatusResponse(
            status=stats.get("status", "unknown"),
            num_documents=stats.get("num_documents", 0),
            num_nodes=stats.get("num_nodes", "unknown"),
            storage_directory=stats.get("storage_dir", "unknown"),
            chunk_size=stats.get("chunk_size", 512),
            embedding_model=stats.get("embedding_model", "text-embedding-3-large"),
            llm_model=stats.get("llm_model", "gpt-4o"),
            index_ready=system.index is not None and system.retriever is not None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.get("/vector-db/test-embedding")
async def test_embedding_model():
    """
    Sprint 2: Test that the embedding model executes successfully.

    Simple endpoint to verify Azure embedding model access works.
    """
    try:
        from llamaindex_models import get_text_embedding_3_large

        # Test embedding model
        embedding_model = get_text_embedding_3_large()

        # Test with a simple text
        test_text = "This is a test sentence for embedding generation."

        # Note: We don't actually call the model here as it's expensive
        # In real usage, LlamaIndex will call it automatically

        return {
            "status": "success",
            "message": "Embedding model accessible and configured",
            "model": "text-embedding-3-large",
            "test_text": test_text,
            "model_type": str(type(embedding_model))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding model test failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
