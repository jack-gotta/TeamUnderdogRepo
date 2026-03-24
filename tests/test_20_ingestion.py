"""Tests for document ingestion pipeline."""

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
