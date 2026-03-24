"""Document ingestion and embedding pipeline for RAG system.

This module handles:
- Loading documents from Hugging Face datasets
- Generating embeddings using Azure OpenAI models
- Storing embeddings in a local LlamaIndex vector store
- Querying the vector store
"""

from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding

from llamaindex_models import get_embedding_model


DEFAULT_PASSAGES_SOURCE = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"


def load_documents_from_parquet(source: str, limit: int | None = None) -> list[Document]:
    """Load documents from a parquet source.

    Args:
        source: Parquet source URI or path.
        limit: Optional number of rows to load.

    Returns:
        List of LlamaIndex Document objects.
    """
    import pandas as pd

    df = pd.read_parquet(source)
    if limit is not None:
        df = df.head(limit)

    text_column = None
    preferred_columns = ["text", "passage", "content", "body"]
    for col in preferred_columns:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        string_columns = [c for c in df.columns if df[c].dtype == "object"]
        if not string_columns:
            raise ValueError("No text-like column found in source data")
        text_column = string_columns[0]

    documents: list[Document] = []
    for _, row in df.iterrows():
        text = str(row[text_column])
        metadata: dict[str, Any] = {}
        if "id" in df.columns:
            metadata["id"] = row["id"]
        if "title" in df.columns:
            metadata["title"] = row["title"]

        documents.append(Document(text=text, metadata=metadata))

    return documents


class VectorStoreManager:
    """Manages vector store lifecycle and operations."""

    def __init__(self, persist_dir: Path | None = None, embedding_model: BaseEmbedding | None = None):
        """Initialize vector store manager.
        
        Args:
            persist_dir: Optional directory to persist the vector store.
            embedding_model: Optional embedding model. If not provided, will load Azure model.
        """
        self.persist_dir = persist_dir or Path.cwd() / ".vector_store"
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        self.index: VectorStoreIndex | None = None
        self._embedding_model = embedding_model

    @property
    def embedding_model(self):
        """Lazy-load embedding model on first access."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def create_or_load_index(self) -> VectorStoreIndex:
        """Create a new vector store index or load an existing one.
        
        Returns:
            VectorStoreIndex instance.
        """
        if self.index is None:
            # Create new index with embedding model
            self.index = VectorStoreIndex(
                nodes=[],
                embed_model=self.embedding_model,
            )
        return self.index

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to ingest.
            
        Returns:
            Number of documents added.
        """
        if self.index is None:
            self.create_or_load_index()

        nodes = [
            TextNode(text=doc.get_content())
            for doc in documents
        ]
        self.index.insert_nodes(nodes)
        return len(nodes)

    def ingest_from_source(self, source: str = DEFAULT_PASSAGES_SOURCE, limit: int | None = None) -> int:
        """Load documents from a source and add them to the vector store.

        Args:
            source: Source path or URI to parquet data.
            limit: Optional row limit for partial ingestion.

        Returns:
            Number of documents ingested.
        """
        documents = load_documents_from_parquet(source=source, limit=limit)
        return self.add_documents(documents)

    def get_status(self) -> dict[str, Any]:
        """Get current vector store status.
        
        Returns:
            Dictionary with status information including document count.
        """
        if self.index is None:
            return {
                "initialized": False,
                "document_count": 0,
                "index_info": None,
            }

        doc_count = len(self.index.docstore.docs)
        return {
            "initialized": True,
            "document_count": doc_count,
            "index_info": "LlamaIndex VectorStoreIndex",
            "persist_dir": str(self.persist_dir),
        }

    def query(self, query_str: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Query the vector store for similar documents.
        
        Args:
            query_str: Query string.
            top_k: Number of top results to return.
            
        Returns:
            List of retrieved document results.
        """
        if self.index is None:
            return []

        retriever = self.index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query_str)

        return [
            {
                "text": node.get_content(),
                "score": node.score,
            }
            for node in results
        ]
