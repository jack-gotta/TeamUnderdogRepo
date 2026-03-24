from fastapi.testclient import TestClient
import main

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


def test_auth_status_endpoint_reports_authenticated(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_ailab_auth_status",
        lambda: {
            "authenticated": True,
            "auth_source": "env_token",
            "scope": "api://ailab/Model.Access",
            "error": None,
        },
    )

    response = client.get("/auth/status")

    assert response.status_code == 200
    assert response.json()["authenticated"] is True
    assert response.json()["auth_source"] == "env_token"


def test_auth_status_endpoint_reports_unauthenticated(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_ailab_auth_status",
        lambda: {
            "authenticated": False,
            "auth_source": None,
            "scope": "api://ailab/Model.Access",
            "error": "No credential available",
        },
    )

    response = client.get("/auth/status")

    assert response.status_code == 200
    assert response.json()["authenticated"] is False
    assert response.json()["error"] == "No credential available"
