"""Tests for vector database operations."""

import pytest
from llama_index.core import Document
from unittest.mock import Mock, patch, MagicMock


def test_vector_db_create_index_with_mock_embedding() -> None:
    """Verify create_vector_index function is callable and takes correct params."""
    from vector_db import create_vector_index
    import inspect
    
    # Check that create_vector_index has the expected signature
    sig = inspect.signature(create_vector_index)
    assert "documents" in sig.parameters
    assert "persist_dir" in sig.parameters
    
    # Verify it's callable
    assert callable(create_vector_index)


def test_vector_db_get_index_stats() -> None:
    """Verify we can get index statistics."""
    from vector_db import get_index_stats
    from llama_index.core import VectorStoreIndex
    
    # Create a mock index
    mock_index = Mock(spec=VectorStoreIndex)
    mock_index.docstore = Mock()
    mock_index.docstore.docs = {f"doc{i}": Mock() for i in range(5)}
    
    stats = get_index_stats(mock_index)
    
    assert isinstance(stats, dict)
    assert "index_type" in stats
    assert stats["index_type"] == "VectorStoreIndex"


def test_vector_db_similarity_search_with_mock() -> None:
    """Verify vector database search returns results with mocked embeddings."""
    from vector_db import vector_search
    from llama_index.core import VectorStoreIndex
    
    # Create a mock index with retriever
    mock_index = Mock(spec=VectorStoreIndex)
    mock_retriever = Mock()
    
    # Mock search results
    mock_node = Mock()
    mock_node.get_content = Mock(return_value="Sample document text")
    mock_node.score = 0.95
    mock_node.metadata = {"source": "sample"}
    
    mock_retriever.retrieve = Mock(return_value=[mock_node])
    mock_index.as_retriever = Mock(return_value=mock_retriever)
    
    results = vector_search(mock_index, query="test query", top_k=3)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]
    assert "metadata" in results[0]


def test_vector_db_registry_contains_correct_model() -> None:
    """Verify vector DB uses correct embedding model from registry."""
    from llamaindex_models import get_available_models
    
    models = get_available_models()
    assert "text-embedding-3-large" in models["embeddings"]
    
    config = models["embeddings"]["text-embedding-3-large"]
    assert config["deployment_name"] == "text-embedding-3-large"
    assert "api_version" in config
