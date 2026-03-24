"""FastAPI app for Mini Wikipedia RAG system."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

# Global state for vector index (loaded once)
_vector_index = None
STATIC_DIR = Path(__file__).resolve().parent / "static"


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


class PromptResponse(QueryResponse):
    prompt: str


class AnswerResponse(PromptResponse):
    answer: str


class EvaluationItem(BaseModel):
    query: str
    expected_answer: str
    generated_answer: str
    score: float


class EvaluationSummary(BaseModel):
    example_count: int
    average_score: float
    passed_count: int
    results: List[EvaluationItem]


app = FastAPI(
    title="Mini Wikipedia RAG API",
    version="0.1.0",
    description="Baseline API endpoints for service health and metadata.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Mini Wikipedia RAG API"}


@app.get("/app")
def frontend_app() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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


@app.post("/rag/prompt", response_model=PromptResponse)
def rag_prompt(body: SearchQuery) -> PromptResponse:
    """Construct an augmented prompt from a query and retrieved documents."""
    global _vector_index

    if _vector_index is None:
        raise HTTPException(
            status_code=400,
            detail="Vector index not initialized. Call /ingest first.",
        )

    from rag_pipeline import prepare_prompt_response

    result = prepare_prompt_response(index=_vector_index, query=body.query, top_k=body.top_k)
    return PromptResponse(
        query=result["query"],
        top_k=result["top_k"],
        query_embedding_dimension=result["query_embedding_dimension"],
        documents=[SearchResult(**document) for document in result["documents"]],
        prompt=result["prompt"],
    )


@app.post("/rag/answer", response_model=AnswerResponse)
def rag_answer(body: SearchQuery) -> AnswerResponse:
    """Generate a grounded answer from retrieved documents and GPT-4o."""
    global _vector_index

    if _vector_index is None:
        raise HTTPException(
            status_code=400,
            detail="Vector index not initialized. Call /ingest first.",
        )

    from rag_pipeline import answer_user_query

    result = answer_user_query(index=_vector_index, query=body.query, top_k=body.top_k)
    return AnswerResponse(
        query=result["query"],
        top_k=result["top_k"],
        query_embedding_dimension=result["query_embedding_dimension"],
        documents=[SearchResult(**document) for document in result["documents"]],
        prompt=result["prompt"],
        answer=result["answer"],
    )


@app.get("/rag/evaluate", response_model=EvaluationSummary)
def rag_evaluate(limit: int = 3, top_k: int = 3) -> EvaluationSummary:
    """Evaluate the RAG answer flow against reference question-answer pairs."""
    global _vector_index

    if _vector_index is None:
        raise HTTPException(
            status_code=400,
            detail="Vector index not initialized. Call /ingest first.",
        )

    from ingestion import load_huggingface_test_questions
    from rag_pipeline import evaluate_rag_pipeline

    examples = load_huggingface_test_questions(count=limit)
    result = evaluate_rag_pipeline(index=_vector_index, evaluation_examples=examples, top_k=top_k)

    return EvaluationSummary(
        example_count=result["example_count"],
        average_score=result["average_score"],
        passed_count=result["passed_count"],
        results=[EvaluationItem(**item) for item in result["results"]],
    )
