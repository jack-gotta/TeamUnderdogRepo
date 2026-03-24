"""
FastAPI server for the Wikipedia RAG system.

This server provides REST API endpoints for the RAG functionality,
including health checks, data ingestion, and question answering.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import our model access layer
try:
    from llamaindex_models import get_gpt4o, get_text_embedding_3_large, get_available_models
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Model imports failed: {e}")
    MODELS_AVAILABLE = False

# Import ingestion pipeline
try:
    from ingestion import get_pipeline
    INGESTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Ingestion pipeline import failed: {e}")
    INGESTION_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Wikipedia RAG System API",
    description="A Retrieval-Augmented Generation system using LlamaIndex with Wikipedia articles",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Response models
class HealthResponse(BaseModel):
    status: str
    message: str
    models_available: bool
    timestamp: str

class EchoRequest(BaseModel):
    message: str
    metadata: Optional[Dict[str, Any]] = None

class EchoResponse(BaseModel):
    echo: str
    received_metadata: Optional[Dict[str, Any]] = None
    timestamp: str

class VectorDBStatusResponse(BaseModel):
    available: bool
    documents_loaded: int = 0
    nodes_created: int = 0
    embeddings_generated: int = 0
    index_created: bool = False
    index_available: bool = False
    last_ingestion: Optional[str] = None
    total_documents: Optional[int] = None
    total_vectors: Optional[int] = None
    timestamp: str

class IngestionRequest(BaseModel):
    max_documents: Optional[int] = None
    force_rebuild: bool = False

class IngestionResponse(BaseModel):
    success: bool
    message: str
    stats: Dict[str, Any]
    timestamp: str

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    include_metadata: bool = True

class RetrievedDocument(BaseModel):
    content: str
    title: Optional[str] = None
    doc_id: Optional[str] = None
    similarity_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[RetrievedDocument]
    total_results: int
    processing_time_ms: float
    timestamp: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the service is running.

    Returns:
        HealthResponse: Status information including model availability
    """
    from datetime import datetime

    # Test model availability
    models_working = False
    if MODELS_AVAILABLE:
        try:
            models = get_available_models()
            models_working = len(models) > 0
        except Exception:
            models_working = False

    return HealthResponse(
        status="healthy" if models_working else "degraded",
        message="Service is running" if models_working else "Service running but models unavailable",
        models_available=models_working,
        timestamp=datetime.utcnow().isoformat()
    )

# Echo endpoint for testing
@app.post("/echo", response_model=EchoResponse)
async def echo_message(request: EchoRequest):
    """
    Simple echo endpoint for testing request/response functionality.

    Args:
        request: EchoRequest with message and optional metadata

    Returns:
        EchoResponse: Echoed message with timestamp
    """
    from datetime import datetime

    return EchoResponse(
        echo=request.message,
        received_metadata=request.metadata,
        timestamp=datetime.utcnow().isoformat()
    )

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "message": "Wikipedia RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "vector_db": {
            "status": "/vectordb/status",
            "ingest": "/vectordb/ingest",
            "info": "/vectordb/info"
        },
        "query": {
            "ask": "/query",
            "search": "/search"
        }
    }

# Model status endpoint
@app.get("/models/status")
async def models_status():
    """
    Get status of available AI models.

    Returns:
        Dict: Model availability and configuration
    """
    if not MODELS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Models not available")

    try:
        available_models = get_available_models()
        return {
            "available": True,
            "models": available_models,
            "endpoint": os.getenv("AILAB_ENDPOINT", "https://ct-enterprisechat-api.azure-api.net/")
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model access failed: {str(e)}")

# Vector database endpoints
@app.get("/vectordb/info")
async def vectordb_info():
    """
    Get information about the vector database system.
    """
    return {
        "description": "Wikipedia RAG Vector Database",
        "embedding_model": "text-embedding-3-large",
        "endpoints": {
            "status": "/vectordb/status",
            "ingest": "/vectordb/ingest",
            "info": "/vectordb/info",
            "query": "/query",
            "search": "/search"
        }
    }

@app.get("/vectordb/status", response_model=VectorDBStatusResponse)
async def vectordb_status():
    """
    Get the current status of the vector database.
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not available")

    try:
        pipeline = get_pipeline()
        stats = pipeline.get_index_stats()

        return VectorDBStatusResponse(
            available=True,
            documents_loaded=stats.get("documents_loaded", 0),
            nodes_created=stats.get("nodes_created", 0),
            embeddings_generated=stats.get("embeddings_generated", 0),
            index_created=stats.get("index_created", False),
            index_available=stats.get("index_available", False),
            last_ingestion=stats.get("last_ingestion"),
            total_documents=stats.get("total_documents"),
            total_vectors=stats.get("total_vectors"),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vector DB status: {str(e)}")

@app.post("/vectordb/ingest", response_model=IngestionResponse)
async def vectordb_ingest(request: IngestionRequest):
    """
    Trigger document ingestion and vector database creation.
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not available")

    try:
        pipeline = get_pipeline()

        # Run ingestion
        success = pipeline.run_ingestion(
            max_docs=request.max_documents,
            force_rebuild=request.force_rebuild
        )

        # Get updated stats
        stats = pipeline.get_index_stats()

        return IngestionResponse(
            success=success,
            message="Ingestion completed successfully" if success else "Ingestion failed",
            stats=stats,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# Query endpoints
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the vector database and get an AI-generated answer with retrieved documents.

    This endpoint provides full RAG (Retrieval-Augmented Generation) functionality:
    1. Converts user query to embedding
    2. Performs similarity search against vector database
    3. Retrieves relevant documents
    4. Generates AI answer based on retrieved context
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not available")

    import time
    start_time = time.time()

    try:
        pipeline = get_pipeline()

        if not pipeline.index:
            # Try to load existing index
            existing_index = pipeline.load_existing_index()
            if not existing_index:
                raise HTTPException(status_code=404, detail="No vector database found. Please run ingestion first.")
            pipeline.index = existing_index

        # Perform the query using the pipeline
        answer = pipeline.query_index(request.query, top_k=request.max_results)

        if not answer:
            raise HTTPException(status_code=500, detail="Query processing failed")

        # Get the retrieved documents from the query engine
        query_engine = pipeline.index.as_query_engine(similarity_top_k=request.max_results)
        response_obj = query_engine.query(request.query)

        retrieved_docs = []
        if hasattr(response_obj, 'source_nodes'):
            for i, node in enumerate(response_obj.source_nodes):
                doc = RetrievedDocument(
                    content=node.text,
                    title=node.metadata.get('title'),
                    doc_id=node.metadata.get('doc_id'),
                    similarity_score=node.score if hasattr(node, 'score') else None,
                    metadata=node.metadata if request.include_metadata else None
                )
                retrieved_docs.append(doc)

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return QueryResponse(
            query=request.query,
            answer=str(answer),
            retrieved_documents=retrieved_docs,
            total_results=len(retrieved_docs),
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        # Re-raise HTTP exceptions (like 404) as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/search")
async def search_documents(request: QueryRequest):
    """
    Search the vector database and return relevant documents without AI generation.

    This endpoint provides document retrieval functionality:
    1. Converts user query to embedding
    2. Performs similarity search against vector database
    3. Returns ranked list of relevant documents
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not available")

    import time
    start_time = time.time()

    try:
        pipeline = get_pipeline()

        if not pipeline.index:
            # Try to load existing index
            existing_index = pipeline.load_existing_index()
            if not existing_index:
                raise HTTPException(status_code=404, detail="No vector database found. Please run ingestion first.")
            pipeline.index = existing_index

        # Create retriever for similarity search
        retriever = pipeline.index.as_retriever(similarity_top_k=request.max_results)

        # Perform similarity search
        retrieved_nodes = retriever.retrieve(request.query)

        retrieved_docs = []
        for node in retrieved_nodes:
            doc = RetrievedDocument(
                content=node.text,
                title=node.metadata.get('title'),
                doc_id=node.metadata.get('doc_id'),
                similarity_score=node.score if hasattr(node, 'score') else None,
                metadata=node.metadata if request.include_metadata else None
            )
            retrieved_docs.append(doc)

        processing_time = (time.time() - start_time) * 1000

        return {
            "query": request.query,
            "retrieved_documents": retrieved_docs,
            "total_results": len(retrieved_docs),
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        # Re-raise HTTP exceptions (like 404) as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run with: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)