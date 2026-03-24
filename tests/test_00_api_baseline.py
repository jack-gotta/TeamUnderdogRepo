from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Mini Wikipedia RAG API"}


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint() -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {"version": "0.1.0"}


def test_openapi_endpoint_available() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Mini Wikipedia RAG API"


def test_echo_endpoint() -> None:
    response = client.post("/echo", json={"message": "hello world"})

    assert response.status_code == 200
    assert response.json() == {"echo": "hello world"}


def test_docs_endpoint_available() -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_frontend_shell_endpoint_available() -> None:
    response = client.get("/app")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Query the retrieval layer" in response.text


def test_frontend_static_javascript_available() -> None:
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "text/javascript" in response.headers["content-type"] or "application/javascript" in response.headers["content-type"]
