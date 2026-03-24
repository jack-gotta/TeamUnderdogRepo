"""
Sprint 1 Tests - Basic FastAPI Infrastructure

Tests for baseline endpoints:
- Health check
- Echo endpoint
- Basic server functionality
"""

import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, 'src')

from main import app

# Create test client
client = TestClient(app)


class TestSprintOneEndpoints:
    """Test Sprint 1 baseline endpoints."""

    def test_root_endpoint(self):
        """Test the root endpoint returns basic info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "python_version" in data
        assert "dependencies_loaded" in data

        # Should be healthy for Sprint 1
        assert data["status"] == "healthy"

    def test_echo_post_endpoint(self):
        """Test the POST echo endpoint."""
        test_message = "Hello, Sprint 1!"
        response = client.post(
            "/echo",
            json={"message": test_message}
        )
        assert response.status_code == 200
        data = response.json()
        assert "echo" in data
        assert "received_at" in data
        assert data["echo"] == test_message

    def test_echo_get_endpoint(self):
        """Test the GET echo endpoint."""
        test_message = "test-message-123"
        response = client.get(f"/echo/{test_message}")
        assert response.status_code == 200
        data = response.json()
        assert "echo" in data
        assert "received_at" in data
        assert "method" in data
        assert data["echo"] == test_message
        assert data["method"] == "GET"

    def test_docs_endpoint_accessible(self):
        """Test that the auto-generated docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        # FastAPI docs return HTML
        assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_spec_available(self):
        """Test that the OpenAPI spec is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


class TestBasicFunctionality:
    """Test basic server functionality."""

    def test_server_can_handle_multiple_requests(self):
        """Test that server can handle multiple requests without issues."""
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200

    def test_invalid_endpoint_returns_404(self):
        """Test that invalid endpoints return 404."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_echo_endpoint_validates_input(self):
        """Test that echo endpoint validates input properly."""
        # Valid input
        response = client.post("/echo", json={"message": "valid"})
        assert response.status_code == 200

        # Invalid input (missing message field)
        response = client.post("/echo", json={"wrong_field": "invalid"})
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
