"""FastAPI app for Mini Wikipedia RAG system."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

# Global state for vector index (loaded once)
_vector_index = None


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    echo: str


class VectorDbStats(BaseModel):
    document_count: int
    index_ready: bool
    persist_dir: Optional[str] = None


class SearchQuery(BaseModel):
    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    query: str
    top_k: int
    query_embedding_dimension: int
    documents: List[SearchResult]


app = FastAPI(
    title="Mini Wikipedia RAG API",
    version="0.1.0",
    description="Baseline API endpoints for service health and metadata.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Mini Wikipedia RAG API"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": app.version}


@app.post("/echo", response_model=EchoResponse)
def echo(body: EchoRequest) -> EchoResponse:
    return EchoResponse(echo=body.message)


@app.post("/ingest")
def ingest(document_count: int = 10) -> Dict[str, Any]:
    """Ingest sample documents and build vector index.
    
    Args:
        document_count: Number of documents to ingest.
        
    Returns:
        Dictionary with ingestion status and index stats.
    """
    global _vector_index
    
    from ingestion import load_sample_documents
    from vector_db import create_vector_index
    
    # Load documents
    documents = load_sample_documents(count=document_count)
    
    # Create index
    _vector_index = create_vector_index(documents=documents)
    
    return {
        "status": "success",
        "documents_ingested": len(documents),
        "index_ready": _vector_index is not None,
    }


@app.get("/vector-db/stats", response_model=VectorDbStats)
def vector_db_stats() -> VectorDbStats:
    """Get vector database statistics.
    
    Returns:
        VectorDbStats with index information.
    """
    global _vector_index
    
    if _vector_index is None:
        return VectorDbStats(
            document_count=0,
            index_ready=False,
        )
    
    from vector_db import get_index_stats
    
    stats = get_index_stats(_vector_index)
    
    return VectorDbStats(
        document_count=stats.get("document_count", 0),
        index_ready=True,
    )


@app.post("/vector-db/search")
def vector_db_search(body: SearchQuery) -> List[SearchResult]:
    """Search the vector database for similar documents.
    
    Args:
        body: SearchQuery with query text and top_k parameter.
        
    Returns:
        List of SearchResult objects.
    """
    global _vector_index
    
    if _vector_index is None:
        raise HTTPException(
            status_code=400,
            detail="Vector index not initialized. Call /ingest first.",
        )
    
    from vector_db import vector_search
    
    results = vector_search(_vector_index, body.query, top_k=body.top_k)
    
    return [SearchResult(**result) for result in results]


@app.post("/query", response_model=QueryResponse)
def query_documents(body: SearchQuery) -> QueryResponse:
    """Accept a user query and return retrieved documents.

    Args:
        body: SearchQuery with query text and top_k parameter.

    Returns:
        QueryResponse with embedding metadata and retrieved documents.
    """
    global _vector_index

    if _vector_index is None:
        raise HTTPException(
            status_code=400,
            detail="Vector index not initialized. Call /ingest first.",
        )

    from vector_db import retrieve_relevant_documents

    result = retrieve_relevant_documents(
        index=_vector_index,
        query=body.query,
        top_k=body.top_k,
    )

    return QueryResponse(
        query=result["query"],
        top_k=result["top_k"],
        query_embedding_dimension=result["query_embedding_dimension"],
        documents=[SearchResult(**document) for document in result["documents"]],
    )
