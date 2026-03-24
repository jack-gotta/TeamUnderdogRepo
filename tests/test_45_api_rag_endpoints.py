"""Tests for Sprint 4 RAG API endpoints."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_rag_prompt_endpoint_requires_index() -> None:
    """Verify /rag/prompt errors when the vector index is not ready."""
    import api

    api._vector_index = None

    response = client.post("/rag/prompt", json={"query": "What is Python?"})

    assert response.status_code == 400
    assert "not initialized" in response.json()["detail"].lower()


def test_rag_prompt_endpoint_returns_prompt_and_documents() -> None:
    """Verify /rag/prompt returns the constructed prompt and retrieved docs."""
    import api

    api._vector_index = Mock()

    with patch('rag_pipeline.prepare_prompt_response') as mock_prepare:
        mock_prepare.return_value = {
            "query": "What is Python?",
            "top_k": 2,
            "query_embedding_dimension": 1536,
            "documents": [
                {"text": "Python context", "score": 0.9, "metadata": {"title": "Python"}}
            ],
            "prompt": "Prompt text",
        }

        response = client.post("/rag/prompt", json={"query": "What is Python?", "top_k": 2})

        assert response.status_code == 200
        data = response.json()
        assert data["prompt"] == "Prompt text"
        assert len(data["documents"]) == 1

    api._vector_index = None


def test_rag_answer_endpoint_returns_answer() -> None:
    """Verify /rag/answer returns answer text plus prompt and documents."""
    import api

    api._vector_index = Mock()

    with patch('rag_pipeline.answer_user_query') as mock_answer:
        mock_answer.return_value = {
            "query": "What is Python?",
            "top_k": 2,
            "query_embedding_dimension": 1536,
            "documents": [
                {"text": "Python context", "score": 0.9, "metadata": {"title": "Python"}}
            ],
            "prompt": "Prompt text",
            "answer": "Python is a programming language.",
        }

        response = client.post("/rag/answer", json={"query": "What is Python?", "top_k": 2})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Python is a programming language."
        assert data["prompt"] == "Prompt text"
        assert data["query_embedding_dimension"] == 1536

    api._vector_index = None


def test_rag_evaluate_endpoint_returns_summary() -> None:
    """Verify /rag/evaluate returns aggregate evaluation metrics."""
    import api

    api._vector_index = Mock()

    with patch('ingestion.load_huggingface_test_questions') as mock_load_examples:
        with patch('rag_pipeline.evaluate_rag_pipeline') as mock_evaluate:
            mock_load_examples.return_value = [
                {"query": "What is Python?", "expected_answer": "Python is a programming language."}
            ]
            mock_evaluate.return_value = {
                "example_count": 1,
                "average_score": 0.75,
                "passed_count": 1,
                "results": [
                    {
                        "query": "What is Python?",
                        "expected_answer": "Python is a programming language.",
                        "generated_answer": "Python is a programming language.",
                        "score": 0.75,
                    }
                ],
            }

            response = client.get("/rag/evaluate?limit=1&top_k=2")

            assert response.status_code == 200
            data = response.json()
            assert data["example_count"] == 1
            assert data["average_score"] == 0.75
            assert data["passed_count"] == 1
            assert len(data["results"]) == 1

    api._vector_index = None