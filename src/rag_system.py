"""
RAG System for Wikipedia Question Answering.

This module implements a complete RAG pipeline that:
1. Loads Wikipedia passages from HuggingFace datasets
2. Generates embeddings using Azure OpenAI's text-embedding-3-large
3. Stores vectors in a local LlamaIndex vector database
4. Retrieves relevant documents based on semantic similarity
5. Generates answers using Azure OpenAI's GPT-4o model
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import QueryBundle
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

from llamaindex_models import get_gpt4o, get_text_embedding_3_large


class WikipediaRAGSystem:
    """A complete RAG system for Wikipedia question answering."""

    def __init__(self, storage_dir: str = "./rag_storage", chunk_size: int = 512):
        """Initialize the RAG system.

        Args:
            storage_dir: Directory to store the vector database and indices
            chunk_size: Size of text chunks for processing
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size

        # Configure LlamaIndex settings with our controlled models
        Settings.llm = get_gpt4o(temperature=0.1, max_tokens=500)
        Settings.embed_model = get_text_embedding_3_large()

        # Initialize storage components
        self._init_storage()

        # RAG components
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self.retriever: Optional[VectorIndexRetriever] = None

    def _init_storage(self):
        """Initialize storage context for persistence."""
        # Check if we have existing storage
        docstore_path = self.storage_dir / "docstore.json"
        index_store_path = self.storage_dir / "index_store.json"
        vector_store_path = self.storage_dir / "vector_store.json"

        if all(p.exists() for p in [docstore_path, index_store_path, vector_store_path]):
            # Load existing storage
            self.storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(str(self.storage_dir)),
                vector_store=SimpleVectorStore.from_persist_dir(str(self.storage_dir)),
                index_store=SimpleIndexStore.from_persist_dir(str(self.storage_dir))
            )
        else:
            # Create new storage
            self.storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                vector_store=SimpleVectorStore(),
                index_store=SimpleIndexStore()
            )

    def load_wikipedia_data(self, dataset_url: str = None) -> List[Document]:
        """Load Wikipedia passages from HuggingFace dataset.

        Args:
            dataset_url: URL to the HuggingFace dataset parquet file

        Returns:
            List of LlamaIndex Document objects
        """
        if dataset_url is None:
            dataset_url = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"

        print(f"Loading Wikipedia data from: {dataset_url}")

        # Load the parquet file
        df = pd.read_parquet(dataset_url)
        print(f"Loaded {len(df)} passages from Wikipedia dataset")

        # Convert to LlamaIndex Documents
        documents = []
        for idx, row in df.iterrows():
            # Create document with passage text and metadata
            doc = Document(
                text=row['passage'],
                metadata={
                    'passage_id': row.get('id', idx),
                    'title': row.get('title', 'Unknown'),
                    'source': 'wikipedia'
                }
            )
            documents.append(doc)

        print(f"Created {len(documents)} documents for indexing")
        return documents

    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> VectorStoreIndex:
        """Build or load the vector index from documents.

        Args:
            documents: List of documents to index
            force_rebuild: Whether to rebuild index even if it exists

        Returns:
            The vector store index
        """
        index_path = self.storage_dir / "index_store.json"

        if not force_rebuild and index_path.exists() and len(self.storage_context.docstore.docs) > 0:
            print("Loading existing vector index...")
            self.index = VectorStoreIndex.from_documents(
                [], storage_context=self.storage_context
            )
            # Load from storage
            self.index = VectorStoreIndex(
                nodes=[], storage_context=self.storage_context
            )
            print(f"Loaded existing index with {len(self.storage_context.docstore.docs)} documents")
        else:
            print("Building new vector index...")

            # Configure text splitter
            text_splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=50
            )

            # Create index with documents
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                transformations=[text_splitter],
                show_progress=True
            )

            # Persist the index
            self.index.storage_context.persist(str(self.storage_dir))
            print(f"Built and persisted index with {len(documents)} documents")

        return self.index

    def setup_retriever(self, similarity_top_k: int = 5) -> VectorIndexRetriever:
        """Setup the document retriever.

        Args:
            similarity_top_k: Number of top similar documents to retrieve

        Returns:
            Configured retriever
        """
        if self.index is None:
            raise ValueError("Index must be built before setting up retriever")

        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )
        print(f"Setup retriever to return top {similarity_top_k} matches")
        return self.retriever

    def setup_query_engine(self) -> RetrieverQueryEngine:
        """Setup the query engine for RAG.

        Returns:
            Configured query engine
        """
        if self.retriever is None:
            raise ValueError("Retriever must be setup before creating query engine")

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            llm=Settings.llm
        )
        print("Setup query engine for RAG pipeline")
        return self.query_engine

    def retrieve_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: The user query
            top_k: Number of documents to retrieve (uses retriever default if None)

        Returns:
            List of retrieved document information
        """
        if self.retriever is None:
            raise ValueError("Retriever must be setup before retrieving documents")

        # Temporarily update top_k if specified
        original_top_k = self.retriever.similarity_top_k
        if top_k is not None:
            self.retriever.similarity_top_k = top_k

        try:
            # Retrieve nodes
            query_bundle = QueryBundle(query_str=query)
            nodes = self.retriever.retrieve(query_bundle)

            # Format results
            results = []
            for i, node in enumerate(nodes):
                result = {
                    'rank': i + 1,
                    'score': getattr(node, 'score', None),
                    'text': node.text,
                    'metadata': node.metadata,
                    'node_id': node.node_id
                }
                results.append(result)

            return results

        finally:
            # Restore original top_k
            self.retriever.similarity_top_k = original_top_k

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate an answer using the full RAG pipeline.

        Args:
            query: The user question

        Returns:
            Dictionary with answer and metadata
        """
        if self.query_engine is None:
            raise ValueError("Query engine must be setup before generating answers")

        print(f"Generating answer for: {query}")

        # Get response from query engine
        response = self.query_engine.query(query)

        # Extract source information
        source_info = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                source_info.append({
                    'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'metadata': node.metadata,
                    'score': getattr(node, 'score', None)
                })

        return {
            'query': query,
            'answer': str(response),
            'sources': source_info,
            'num_sources': len(source_info)
        }

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index.

        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {'status': 'No index available'}

        # Get document count from storage
        num_docs = len(self.storage_context.docstore.docs)

        # Try to get node count
        try:
            # This is a rough approximation
            num_nodes = len(self.index.storage_context.vector_store.data)
        except:
            num_nodes = "Unknown"

        return {
            'status': 'Index available',
            'num_documents': num_docs,
            'num_nodes': num_nodes,
            'storage_dir': str(self.storage_dir),
            'chunk_size': self.chunk_size,
            'embedding_model': 'text-embedding-3-large',
            'llm_model': 'gpt-4o'
        }


def create_rag_system(storage_dir: str = "./rag_storage") -> WikipediaRAGSystem:
    """Factory function to create a RAG system.

    Args:
        storage_dir: Directory for persistent storage

    Returns:
        Configured WikipediaRAGSystem instance
    """
    return WikipediaRAGSystem(storage_dir=storage_dir)
