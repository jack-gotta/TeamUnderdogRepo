"""Integration tests for vector database with Azure authentication.

These tests require Azure OpenAI authentication to be set up:
  azd auth login --scope api://ailab/Model.Access

Run with: uv run pytest tests/test_50_vector_db_integration.py
"""

import pytest
import os


# Skip all tests in this file if AZURE_AUTH_AVAILABLE is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("AZURE_AUTH_AVAILABLE"),
    reason="Azure authentication not available. Set AZURE_AUTH_AVAILABLE=1 to run."
)


def test_vector_db_create_real_index() -> None:
    """Integration test: Create real vector index with Azure embeddings."""
    from vector_db import create_vector_index
    from ingestion import load_sample_documents
    
    docs = load_sample_documents(count=3)
    index = create_vector_index(documents=docs)
    
    assert index is not None
    assert hasattr(index, 'as_retriever')


def test_vector_db_real_similarity_search() -> None:
    """Integration test: Perform real similarity search."""
    from vector_db import create_vector_index, vector_search
    from ingestion import load_sample_documents
    
    docs = load_sample_documents(count=5)
    index = create_vector_index(documents=docs)
    
    results = vector_search(index, query="Python programming language", top_k=2)
    
    assert isinstance(results, list)
    assert len(results) <= 2
    # Real search should return documents
    assert len(results) > 0


def test_query_to_retrieval_flow_real() -> None:
    """Integration test: Embed a query and retrieve relevant documents end-to-end."""
    from vector_db import create_vector_index, retrieve_relevant_documents
    from ingestion import load_sample_documents

    docs = load_sample_documents(count=5)
    index = create_vector_index(documents=docs)

    result = retrieve_relevant_documents(index, query="Python programming language", top_k=2)

    assert result["query"] == "Python programming language"
    assert result["top_k"] == 2
    assert result["query_embedding_dimension"] > 0
    assert len(result["documents"]) > 0


def test_vector_db_persist_real() -> None:
    """Integration test: Persist and reload real index."""
    import tempfile
    import os
    from vector_db import create_vector_index, load_vector_index
    from ingestion import load_sample_documents
    
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = load_sample_documents(count=3)
        
        # Create and persist
        index_path = os.path.join(tmpdir, "test_index")
        create_vector_index(documents=docs, persist_dir=index_path)
        
        # Reload
        loaded_index = load_vector_index(persist_dir=index_path)
        assert loaded_index is not None
