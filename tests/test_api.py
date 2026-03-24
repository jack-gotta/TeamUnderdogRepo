from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_echo_endpoint() -> None:
    payload = {"message": "hello"}

    response = client.post("/echo", json=payload)

    assert response.status_code == 200
    assert response.json() == payload
