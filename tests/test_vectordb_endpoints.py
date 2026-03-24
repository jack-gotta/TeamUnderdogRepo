"""
Tests for vector database API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import app

client = TestClient(app)


class TestVectorDBEndpoints:
    """Test cases for vector database API endpoints."""

    def test_vectordb_info_endpoint(self):
        """Test the vector database info endpoint."""
        response = client.get("/vectordb/info")

        assert response.status_code == 200
        data = response.json()

        assert "description" in data
        assert "embedding_model" in data
        assert data["embedding_model"] == "text-embedding-3-large"
        assert "endpoints" in data
        assert "status" in data["endpoints"]
        assert "ingest" in data["endpoints"]
        assert "info" in data["endpoints"]

    @patch('main.INGESTION_AVAILABLE', True)
    @patch('main.get_pipeline')
    def test_vectordb_status_success(self, mock_get_pipeline):
        """Test vector database status endpoint success."""
        # Mock pipeline and stats
        mock_pipeline = Mock()
        mock_stats = {
            "documents_loaded": 100,
            "nodes_created": 250,
            "embeddings_generated": 250,
            "index_created": True,
            "last_ingestion": "2026-03-24T10:00:00",
            "index_available": True,
            "total_documents": 100,
            "total_vectors": 250
        }
        mock_pipeline.get_index_stats.return_value = mock_stats
        mock_get_pipeline.return_value = mock_pipeline

        response = client.get("/vectordb/status")

        assert response.status_code == 200
        data = response.json()

        assert data["available"] is True
        assert data["documents_loaded"] == 100
        assert data["nodes_created"] == 250
        assert data["embeddings_generated"] == 250
        assert data["index_created"] is True
        assert data["index_available"] is True
        assert "timestamp" in data

    @patch('main.INGESTION_AVAILABLE', False)
    def test_vectordb_status_unavailable(self):
        """Test vector database status when ingestion unavailable."""
        response = client.get("/vectordb/status")

        assert response.status_code == 503
        assert "Ingestion pipeline not available" in response.json()["detail"]

    @patch('main.INGESTION_AVAILABLE', True)
    @patch('main.get_pipeline')
    def test_vectordb_status_error(self, mock_get_pipeline):
        """Test vector database status endpoint error handling."""
        mock_get_pipeline.side_effect = Exception("Pipeline error")

        response = client.get("/vectordb/status")

        assert response.status_code == 500
        assert "Failed to get vector DB status" in response.json()["detail"]

    @patch('main.INGESTION_AVAILABLE', True)
    @patch('main.get_pipeline')
    def test_vectordb_ingest_success(self, mock_get_pipeline):
        """Test vector database ingestion endpoint success."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.run_ingestion.return_value = True
        mock_stats = {
            "documents_loaded": 50,
            "nodes_created": 125,
            "embeddings_generated": 125,
            "index_created": True,
            "index_available": True
        }
        mock_pipeline.get_index_stats.return_value = mock_stats
        mock_get_pipeline.return_value = mock_pipeline

        request_data = {
            "max_documents": 50,
            "force_rebuild": True
        }

        response = client.post("/vectordb/ingest", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "Ingestion completed successfully" in data["message"]
        assert data["stats"]["documents_loaded"] == 50
        assert "timestamp" in data

        mock_pipeline.run_ingestion.assert_called_once_with(
            max_docs=50,
            force_rebuild=True
        )

    @patch('main.INGESTION_AVAILABLE', True)
    @patch('main.get_pipeline')
    def test_vectordb_ingest_failure(self, mock_get_pipeline):
        """Test vector database ingestion endpoint failure."""
        # Mock pipeline with failed ingestion
        mock_pipeline = Mock()
        mock_pipeline.run_ingestion.return_value = False
        mock_stats = {"index_created": False}
        mock_pipeline.get_index_stats.return_value = mock_stats
        mock_get_pipeline.return_value = mock_pipeline

        request_data = {
            "max_documents": 10,
            "force_rebuild": False
        }

        response = client.post("/vectordb/ingest", json=request_data)

        assert response.status_code == 200  # Still 200, but success=False
        data = response.json()

        assert data["success"] is False
        assert "Ingestion failed" in data["message"]

    @patch('main.INGESTION_AVAILABLE', False)
    def test_vectordb_ingest_unavailable(self):
        """Test vector database ingestion when pipeline unavailable."""
        request_data = {"max_documents": 10}

        response = client.post("/vectordb/ingest", json=request_data)

        assert response.status_code == 503
        assert "Ingestion pipeline not available" in response.json()["detail"]

    @patch('main.INGESTION_AVAILABLE', True)
    @patch('main.get_pipeline')
    def test_vectordb_ingest_exception(self, mock_get_pipeline):
        """Test vector database ingestion endpoint exception handling."""
        mock_get_pipeline.side_effect = Exception("Pipeline crashed")

        request_data = {"max_documents": 10}

        response = client.post("/vectordb/ingest", json=request_data)

        assert response.status_code == 500
        assert "Ingestion failed" in response.json()["detail"]

    def test_vectordb_ingest_default_parameters(self):
        """Test ingestion endpoint with default parameters."""
        with patch('main.INGESTION_AVAILABLE', True), \
             patch('main.get_pipeline') as mock_get_pipeline:

            mock_pipeline = Mock()
            mock_pipeline.run_ingestion.return_value = True
            mock_pipeline.get_index_stats.return_value = {}
            mock_get_pipeline.return_value = mock_pipeline

            # Empty request should use defaults
            response = client.post("/vectordb/ingest", json={})

            assert response.status_code == 200
            mock_pipeline.run_ingestion.assert_called_once_with(
                max_docs=None,
                force_rebuild=False
            )

    def test_root_endpoint_includes_vectordb(self):
        """Test that root endpoint includes vector DB information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "vector_db" in data
        assert "status" in data["vector_db"]
        assert "ingest" in data["vector_db"]
        assert "info" in data["vector_db"]
        assert data["vector_db"]["status"] == "/vectordb/status"
        assert data["vector_db"]["ingest"] == "/vectordb/ingest"
        assert data["vector_db"]["info"] == "/vectordb/info"


class TestVectorDBEndpointValidation:
    """Test request/response validation for vector DB endpoints."""

    def test_ingest_request_validation(self):
        """Test ingestion request parameter validation."""
        with patch('main.INGESTION_AVAILABLE', True), \
             patch('main.get_pipeline') as mock_get_pipeline:

            mock_pipeline = Mock()
            mock_pipeline.run_ingestion.return_value = True
            mock_pipeline.get_index_stats.return_value = {}
            mock_get_pipeline.return_value = mock_pipeline

            # Test with valid parameters
            valid_request = {
                "max_documents": 100,
                "force_rebuild": True
            }
            response = client.post("/vectordb/ingest", json=valid_request)
            assert response.status_code == 200

            # Test with invalid max_documents type
            invalid_request = {
                "max_documents": "not_a_number",
                "force_rebuild": True
            }
            response = client.post("/vectordb/ingest", json=invalid_request)
            assert response.status_code == 422  # Validation error

            # Test with invalid force_rebuild type
            invalid_request = {
                "max_documents": 100,
                "force_rebuild": "not_a_boolean"
            }
            response = client.post("/vectordb/ingest", json=invalid_request)
            assert response.status_code == 422  # Validation error

    def test_status_response_structure(self):
        """Test that status response has correct structure."""
        with patch('main.INGESTION_AVAILABLE', True), \
             patch('main.get_pipeline') as mock_get_pipeline:

            mock_pipeline = Mock()
            expected_stats = {
                "documents_loaded": 100,
                "nodes_created": 200,
                "embeddings_generated": 200,
                "index_created": True,
                "last_ingestion": "2026-03-24T10:00:00",
                "index_available": True,
                "total_documents": 100,
                "total_vectors": 200
            }
            mock_pipeline.get_index_stats.return_value = expected_stats
            mock_get_pipeline.return_value = mock_pipeline

            response = client.get("/vectordb/status")

            assert response.status_code == 200
            data = response.json()

            # Check all required fields are present
            required_fields = [
                "available", "documents_loaded", "nodes_created",
                "embeddings_generated", "index_created", "index_available",
                "timestamp"
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # Check optional fields
            assert data.get("last_ingestion") == "2026-03-24T10:00:00"
            assert data.get("total_documents") == 100
            assert data.get("total_vectors") == 200