"""Tests for vector database API endpoints."""

from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from api import app


client = TestClient(app)


def test_ingest_endpoint_returns_success() -> None:
    """Verify /ingest endpoint returns success."""
    with patch('ingestion.load_sample_documents') as mock_load_docs:
        with patch('vector_db.create_vector_index') as mock_create_index:
            # Setup mocks
            mock_docs = [Mock() for _ in range(5)]
            mock_load_docs.return_value = mock_docs
            
            mock_index = Mock()
            mock_create_index.return_value = mock_index
            
            response = client.post("/ingest?document_count=5")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["documents_ingested"] == 5
            assert data["index_ready"] is True


def test_vector_db_stats_endpoint_no_index() -> None:
    """Verify /vector-db/stats returns empty stats when no index loaded."""
    # Reset global state
    import api
    api._vector_index = None
    
    response = client.get("/vector-db/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] == 0
    assert data["index_ready"] is False


def test_vector_db_stats_endpoint_with_index() -> None:
    """Verify /vector-db/stats returns stats when index is loaded."""
    import api
    
    # Mock index
    mock_index = Mock()
    mock_index.docstore = Mock()
    mock_index.docstore.docs = {"doc1": Mock(), "doc2": Mock()}
    api._vector_index = mock_index
    
    with patch('vector_db.get_index_stats') as mock_get_stats:
        mock_get_stats.return_value = {
            "index_type": "VectorStoreIndex",
            "document_count": 2,
            "has_retriever": True,
        }
        
        response = client.get("/vector-db/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 2
        assert data["index_ready"] is True
    
    # Clean up
    api._vector_index = None


def test_search_endpoint_no_index() -> None:
    """Verify /vector-db/search returns error when no index loaded."""
    import api
    api._vector_index = None
    
    response = client.post("/vector-db/search", json={"query": "test"})
    
    assert response.status_code == 400
    assert "not initialized" in response.json()["detail"].lower()


def test_search_endpoint_with_mock_index() -> None:
    """Verify /vector-db/search returns results when index is loaded."""
    import api
    
    # Mock index with retriever
    mock_index = Mock()
    mock_retriever = Mock()
    
    mock_result = Mock()
    mock_result.get_content = Mock(return_value="Found document")
    mock_result.score = 0.95
    mock_result.metadata = {"source": "test"}
    
    mock_retriever.retrieve = Mock(return_value=[mock_result])
    mock_index.as_retriever = Mock(return_value=mock_retriever)
    api._vector_index = mock_index
    
    with patch('vector_db.vector_search') as mock_vector_search:
        mock_vector_search.return_value = [
            {"text": "Found document", "score": 0.95, "metadata": {"source": "test"}}
        ]
        
        response = client.post("/vector-db/search", json={"query": "test query", "top_k": 1})
        
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1
        assert results[0]["text"] == "Found document"
        assert results[0]["score"] == 0.95
    
    # Clean up
    api._vector_index = None
