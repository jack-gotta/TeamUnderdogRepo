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


def test_embed_query_uses_controlled_embedding_model() -> None:
    """Verify user queries are embedded through the model isolation layer."""
    from vector_db import embed_query

    with patch('llamaindex_models.get_text_embedding_3_large') as mock_get_embedding_model:
        mock_embedding_model = Mock()
        mock_embedding_model.get_query_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding_model.return_value = mock_embedding_model

        embedding = embed_query("What is Python?")

        assert embedding == [0.1, 0.2, 0.3]
        mock_embedding_model.get_query_embedding.assert_called_once_with("What is Python?")


def test_retrieve_relevant_documents_returns_query_metadata() -> None:
    """Verify query retrieval returns embedding metadata and top documents."""
    from vector_db import retrieve_relevant_documents
    from llama_index.core import VectorStoreIndex

    mock_index = Mock(spec=VectorStoreIndex)

    with patch('vector_db.embed_query', return_value=[0.1, 0.2, 0.3, 0.4]) as mock_embed_query:
        with patch('vector_db.vector_search') as mock_vector_search:
            mock_vector_search.return_value = [
                {"text": "Python document", "score": 0.87, "metadata": {"source": "sample"}}
            ]

            result = retrieve_relevant_documents(mock_index, query="What is Python?", top_k=2)

            assert result["query"] == "What is Python?"
            assert result["top_k"] == 2
            assert result["query_embedding_dimension"] == 4
            assert len(result["documents"]) == 1
            mock_embed_query.assert_called_once_with("What is Python?")
            mock_vector_search.assert_called_once_with(index=mock_index, query="What is Python?", top_k=2)


def test_vector_db_registry_contains_correct_model() -> None:
    """Verify vector DB uses correct embedding model from registry."""
    from llamaindex_models import get_available_models
    
    models = get_available_models()
    assert "text-embedding-3-large" in models["embeddings"]
    
    config = models["embeddings"]["text-embedding-3-large"]
    assert config["deployment_name"] == "text-embedding-3-large"
    assert "api_version" in config
