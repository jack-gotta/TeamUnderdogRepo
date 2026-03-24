"""
Basic tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns expected response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Wikipedia RAG System API"
    assert data["version"] == "1.0.0"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_health_endpoint():
    """Test the health endpoint returns status information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data
    assert "models_available" in data
    assert "timestamp" in data
    assert data["status"] in ["healthy", "degraded"]


def test_echo_endpoint():
    """Test the echo endpoint with a simple message."""
    test_message = "Hello, World!"
    response = client.post("/echo", json={"message": test_message})
    assert response.status_code == 200
    data = response.json()
    assert data["echo"] == test_message
    assert "timestamp" in data


def test_echo_endpoint_with_metadata():
    """Test the echo endpoint with metadata."""
    test_message = "Test with metadata"
    test_metadata = {"key1": "value1", "key2": "value2"}
    response = client.post("/echo", json={"message": test_message, "metadata": test_metadata})
    assert response.status_code == 200
    data = response.json()
    assert data["echo"] == test_message
    assert data["received_metadata"] == test_metadata
    assert "timestamp" in data


def test_models_status_endpoint():
    """Test the models status endpoint."""
    response = client.get("/models/status")
    # This endpoint may fail if models aren't available, which is expected
    # We just want to make sure it returns a valid response structure
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "available" in data
        assert "models" in data
        assert "endpoint" in data