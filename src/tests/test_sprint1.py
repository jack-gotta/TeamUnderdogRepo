import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, "src")

from main import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_echo_post_endpoint():
    """Test the POST echo endpoint.""" 
    response = client.post("/echo", json={"message": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["echo"] == "test"

def test_docs_accessible():
    """Test that docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
