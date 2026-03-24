"""Tests for document ingestion pipeline."""

from unittest.mock import Mock, patch

from llama_index.core import Document


def test_ingest_sample_documents() -> None:
    """Verify we can ingest sample documents into a list."""
    from ingestion import load_sample_documents
    
    docs = load_sample_documents(count=5)
    
    assert isinstance(docs, list)
    assert len(docs) >= 5
    assert all(isinstance(doc, Document) for doc in docs)
    assert all(doc.text for doc in docs)


def test_ingest_documents_have_metadata() -> None:
    """Verify ingested documents contain metadata."""
    from ingestion import load_sample_documents
    
    docs = load_sample_documents(count=3)
    
    for doc in docs:
        assert doc.metadata is not None or doc.metadata == {}


def test_ingest_document_text_not_empty() -> None:
    """Verify each document has non-empty text content."""
    from ingestion import load_sample_documents
    
    docs = load_sample_documents(count=5)
    
    for doc in docs:
        assert len(doc.text) > 0
        assert isinstance(doc.text, str)


def test_ingest_consistent_across_calls() -> None:
    """Verify document loading is deterministic."""
    from ingestion import load_sample_documents
    
    docs1 = load_sample_documents(count=3)
    docs2 = load_sample_documents(count=3)
    
    # Same number of documents
    assert len(docs1) == len(docs2)
    # Same texts (for determinism)
    assert [d.text for d in docs1] == [d.text for d in docs2]


def test_load_huggingface_documents_uses_hf_rows_when_available() -> None:
    """Verify HuggingFace document loader maps passage rows into Documents."""
    from ingestion import load_huggingface_documents

    mock_dataset = [
        {"id": 11, "passage": "Passage one."},
        {"id": 12, "passage": "Passage two."},
    ]
    mock_loader = Mock(return_value=mock_dataset)

    with patch("ingestion._get_hf_load_dataset", return_value=mock_loader):
        docs = load_huggingface_documents(count=2)

    assert len(docs) == 2
    assert docs[0].text == "Passage one."
    assert docs[0].metadata["source"] == "huggingface"
    assert docs[0].metadata["id"] == 11


def test_spread_indices_evenly_samples_dataset() -> None:
    """Verify spread index helper selects rows across the corpus, not just the head."""
    from ingestion import _spread_indices

    indices = _spread_indices(total_count=10, desired_count=3)

    assert indices[0] == 0
    assert indices[-1] == 9
    assert indices != [0, 1, 2]


def test_load_huggingface_documents_spreads_selected_rows() -> None:
    """Verify HuggingFace loader samples spread rows when count is smaller than dataset."""
    from ingestion import load_huggingface_documents

    mock_dataset = [{"id": i, "passage": f"Passage {i}"} for i in range(10)]
    mock_loader = Mock(return_value=mock_dataset)

    with patch("ingestion._get_hf_load_dataset", return_value=mock_loader):
        docs = load_huggingface_documents(count=3)

    dataset_indices = [doc.metadata["dataset_index"] for doc in docs]
    assert dataset_indices[0] == 0
    assert dataset_indices[-1] == 9
    assert dataset_indices != [0, 1, 2]


def test_load_huggingface_test_questions_uses_hf_rows_when_available() -> None:
    """Verify HuggingFace test-question loader maps question/answer pairs."""
    from ingestion import load_huggingface_test_questions

    mock_dataset = [
        {"id": 101, "question": "Is sky blue?", "answer": "yes"},
        {"id": 102, "question": "Is fire cold?", "answer": "no"},
    ]
    mock_loader = Mock(return_value=mock_dataset)

    with patch("ingestion._get_hf_load_dataset", return_value=mock_loader):
        examples = load_huggingface_test_questions(count=2)

    assert len(examples) == 2
    assert examples[0]["query"] == "Is sky blue?"
    assert examples[0]["expected_answer"] == "yes"
    assert examples[0]["source"] == "huggingface"
