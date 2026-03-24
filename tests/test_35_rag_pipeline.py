"""Tests for the Sprint 4 RAG pipeline helpers."""

from unittest.mock import Mock, patch


def test_build_augmented_prompt_includes_query_and_context() -> None:
    """Verify prompt construction includes the user query and retrieved context."""
    from rag_pipeline import build_augmented_prompt

    prompt = build_augmented_prompt(
        query="What is Python?",
        retrieved_documents=[
            {
                "text": "Python is a high-level programming language.",
                "score": 0.91,
                "metadata": {"title": "Python"},
            }
        ],
    )

    assert "What is Python?" in prompt
    assert "Python is a high-level programming language." in prompt
    assert "Retrieved Context" in prompt


def test_prepare_prompt_response_wraps_retrieval_result() -> None:
    """Verify prompt preparation preserves retrieval metadata and adds the prompt."""
    from rag_pipeline import prepare_prompt_response

    mock_index = Mock()

    with patch('rag_pipeline.retrieve_relevant_documents') as mock_retrieve:
        mock_retrieve.return_value = {
            "query": "What is Python?",
            "top_k": 2,
            "query_embedding_dimension": 1536,
            "documents": [
                {"text": "Python context", "score": 0.8, "metadata": {"title": "Python"}}
            ],
        }

        result = prepare_prompt_response(index=mock_index, query="What is Python?", top_k=2)

        assert result["query"] == "What is Python?"
        assert result["query_embedding_dimension"] == 1536
        assert "prompt" in result
        assert "Python context" in result["prompt"]


def test_generate_answer_from_prompt_uses_controlled_chat_model() -> None:
    """Verify answer generation goes through the GPT-4o isolation layer."""
    from rag_pipeline import generate_answer_from_prompt

    with patch('rag_pipeline.get_gpt4o') as mock_get_gpt4o:
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "Python is a programming language."
        mock_llm.complete.return_value = mock_response
        mock_get_gpt4o.return_value = mock_llm

        answer = generate_answer_from_prompt("Prompt text")

        assert answer == "Python is a programming language."
        mock_llm.complete.assert_called_once_with("Prompt text")


def test_answer_user_query_returns_answer_and_prompt() -> None:
    """Verify the full query-to-answer helper composes prompt generation and completion."""
    from rag_pipeline import answer_user_query

    mock_index = Mock()

    with patch('rag_pipeline.prepare_prompt_response') as mock_prepare:
        with patch('rag_pipeline.generate_answer_from_prompt') as mock_generate:
            mock_prepare.return_value = {
                "query": "What is Python?",
                "top_k": 2,
                "query_embedding_dimension": 1536,
                "documents": [
                    {"text": "Python context", "score": 0.9, "metadata": {"title": "Python"}}
                ],
                "prompt": "Prompt text",
            }
            mock_generate.return_value = "Python is a programming language."

            result = answer_user_query(index=mock_index, query="What is Python?", top_k=2)

            assert result["answer"] == "Python is a programming language."
            assert result["prompt"] == "Prompt text"


def test_score_generated_answer_returns_overlap_ratio() -> None:
    """Verify lexical evaluation returns a normalized overlap score."""
    from rag_pipeline import score_generated_answer

    score = score_generated_answer(
        expected_answer="Python is a programming language",
        generated_answer="Python is a language",
    )

    assert score == 0.8


def test_evaluate_rag_pipeline_aggregates_results() -> None:
    """Verify evaluation returns aggregate scores over multiple examples."""
    from rag_pipeline import evaluate_rag_pipeline

    mock_index = Mock()
    examples = [
        {"query": "What is Python?", "expected_answer": "Python is a programming language."},
        {"query": "What is ML?", "expected_answer": "Machine learning uses algorithms."},
    ]

    with patch('rag_pipeline.answer_user_query') as mock_answer:
        mock_answer.side_effect = [
            {"answer": "Python is a programming language."},
            {"answer": "Machine learning uses algorithms."},
        ]

        result = evaluate_rag_pipeline(index=mock_index, evaluation_examples=examples, top_k=2)

        assert result["example_count"] == 2
        assert result["average_score"] == 1.0
        assert result["passed_count"] == 2
        assert len(result["results"]) == 2