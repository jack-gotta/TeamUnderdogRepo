"""
Document ingestion pipeline for the Wikipedia RAG system.

This module handles loading documents, processing them into chunks,
generating embeddings, and creating vector indexes.
"""
import os
import pandas as pd
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser

from llamaindex_models import get_text_embedding_3_large, get_gpt4o


class IngestionPipeline:
    """Pipeline for ingesting documents and creating vector indexes."""

    def __init__(self, storage_dir: str = "storage"):
        """
        Initialize the ingestion pipeline.

        Args:
            storage_dir: Directory to store vector index and other data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Model instances (will be set in initialize_models)
        self.embedding_model = None
        self.llm_model = None
        self.index = None

        # Statistics tracking
        self.stats = {
            "documents_loaded": 0,
            "nodes_created": 0,
            "embeddings_generated": 0,
            "index_created": False,
            "last_ingestion": None
        }

    def initialize_models(self) -> bool:
        """
        Initialize the embedding and LLM models.

        Returns:
            True if successful, False otherwise
        """
        try:
            print("Initializing Azure OpenAI models...")
            self.embedding_model = get_text_embedding_3_large()
            self.llm_model = get_gpt4o()

            # Set global LlamaIndex settings
            Settings.embed_model = self.embedding_model
            Settings.llm = self.llm_model

            print("+ Models initialized successfully")
            return True

        except Exception as e:
            print(f"- Model initialization failed: {e}")
            return False

    def load_wikipedia_data(self, max_docs: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Load Wikipedia data from HuggingFace dataset.

        Args:
            max_docs: Maximum number of documents to load

        Returns:
            DataFrame with Wikipedia articles or None if failed
        """
        try:
            print("Loading Wikipedia data from HuggingFace...")
            url = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"
            df = pd.read_parquet(url)

            if max_docs:
                df = df.head(max_docs)

            self.stats["documents_loaded"] = len(df)
            print(f"+ Loaded {len(df)} documents")
            return df

        except Exception as e:
            print(f"- Failed to load Wikipedia data: {e}")
            return None

    def create_documents(self, df: pd.DataFrame, max_docs: Optional[int] = None) -> List[Document]:
        """
        Convert DataFrame rows to LlamaIndex Document objects.

        Args:
            df: DataFrame containing document data
            max_docs: Maximum number of documents to process

        Returns:
            List of Document objects
        """
        documents = []

        # Apply document limit if specified
        if max_docs:
            df = df.head(max_docs)

        # Detect data format and adapt accordingly
        has_passage = 'passage' in df.columns
        has_text = 'text' in df.columns
        has_title = 'title' in df.columns

        print(f"+ Detected data format: passage={has_passage}, text={has_text}, title={has_title}")

        for idx, row in df.iterrows():
            # Extract text content based on available columns
            if has_passage:
                text_content = str(row.get('passage', ''))
            elif has_text:
                text_content = str(row.get('text', ''))
            else:
                # Fallback: use first available string column
                text_content = str(row.iloc[0]) if len(row) > 0 else ''

            # Extract or generate title
            if has_title:
                title_content = str(row.get('title', f'Document_{idx}'))
            else:
                # Generate title from first sentence or first 50 characters of text
                if text_content:
                    # Try to get first sentence
                    first_sentence = text_content.split('.')[0].strip()
                    if len(first_sentence) > 5 and len(first_sentence) <= 100:
                        title_content = first_sentence
                    else:
                        # Use first 50 characters
                        title_content = text_content[:50].strip()
                        if len(title_content) == 50:
                            title_content += "..."
                else:
                    title_content = f'Document_{idx}'

            doc = Document(
                text=text_content,
                metadata={
                    'title': title_content,
                    'doc_id': str(idx),
                    'source': 'wikipedia',
                    'original_index': idx
                }
            )
            documents.append(doc)

        print(f"+ Created {len(documents)} document objects with content")
        return documents

    def create_vector_index(self, documents: List[Document]) -> Optional[VectorStoreIndex]:
        """
        Create a vector index from documents.

        Args:
            documents: List of Document objects

        Returns:
            VectorStoreIndex or None if failed
        """
        try:
            print("Processing documents and creating vector index...")

            # Parse documents into nodes
            parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(documents)
            self.stats["nodes_created"] = len(nodes)
            print(f"+ Created {len(nodes)} text chunks")

            # Create vector index
            print("Generating embeddings and building index...")
            index = VectorStoreIndex(nodes=nodes, show_progress=True)

            # Save index to storage
            index_path = self.storage_dir / "vector_index"
            index.storage_context.persist(persist_dir=str(index_path))

            self.stats["embeddings_generated"] = len(nodes)
            self.stats["index_created"] = True
            self.stats["last_ingestion"] = datetime.now().isoformat()

            print(f"+ Vector index created with {len(nodes)} embeddings")
            print(f"+ Index saved to {index_path}")

            self.index = index
            return index

        except Exception as e:
            print(f"- Vector index creation failed: {e}")
            return None

    def load_existing_index(self) -> Optional[VectorStoreIndex]:
        """
        Load an existing vector index from storage.

        Returns:
            VectorStoreIndex or None if not found
        """
        try:
            index_path = self.storage_dir / "vector_index"
            if not index_path.exists():
                return None

            print(f"Loading existing index from {index_path}...")

            # Recreate storage context and load index
            from llama_index.core import StorageContext, load_index_from_storage

            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
            index = load_index_from_storage(storage_context)

            self.index = index
            print("+ Existing index loaded successfully")
            return index

        except Exception as e:
            print(f"- Failed to load existing index: {e}")
            return None

    def run_ingestion(self, max_docs: Optional[int] = None, force_rebuild: bool = False) -> bool:
        """
        Run the complete ingestion pipeline.

        Args:
            max_docs: Maximum number of documents to process
            force_rebuild: Whether to force rebuild even if index exists

        Returns:
            True if successful, False otherwise
        """
        print(f"Starting ingestion pipeline (max_docs={max_docs}, force_rebuild={force_rebuild})")

        # Initialize models
        if not self.initialize_models():
            return False

        # Check for existing index unless force rebuild
        if not force_rebuild:
            existing_index = self.load_existing_index()
            if existing_index:
                print("+ Using existing vector index")
                return True

        # Load Wikipedia data
        df = self.load_wikipedia_data(max_docs)
        if df is None:
            return False

        # Create documents
        documents = self.create_documents(df, max_docs)
        if not documents:
            print("- No documents created")
            return False

        # Create vector index
        index = self.create_vector_index(documents)
        if index is None:
            return False

        print("+ Ingestion pipeline completed successfully")
        return True

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Dictionary with index statistics
        """
        stats = self.stats.copy()
        stats["index_available"] = self.index is not None
        stats["storage_path"] = str(self.storage_dir)

        if self.index:
            # Add runtime statistics if available
            try:
                stats["total_documents"] = stats.get("documents_loaded", 0)
                stats["total_vectors"] = stats.get("embeddings_generated", 0)
            except Exception:
                pass

        return stats

    def query_index(self, query: str, top_k: int = 5) -> Optional[str]:
        """
        Query the vector index.

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            Query response or None if no index available
        """
        if not self.index:
            return None

        try:
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            print(f"- Query failed: {e}")
            return None


# Global pipeline instance for singleton access
_pipeline_instance = None


def get_pipeline(storage_dir: str = "storage") -> IngestionPipeline:
    """
    Get the global pipeline instance (singleton pattern).

    Args:
        storage_dir: Storage directory for the pipeline

    Returns:
        IngestionPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = IngestionPipeline(storage_dir)
    return _pipeline_instance