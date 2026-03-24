"""Vector database operations using LlamaIndex VectorStoreIndex.

Manages creation, persistence, and queries against the local vector database.
"""

import os
from typing import Any, List, Dict, Optional
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document, BaseNode, TextNode


def create_vector_index(
    documents: List[Document],
    persist_dir: Optional[str] = None,
) -> VectorStoreIndex:
    """Create a VectorStoreIndex from documents using controlled embedding model.
    
    Args:
        documents: List of LlamaIndex Document objects to index.
        persist_dir: Optional directory to persist the index.
        
    Returns:
        VectorStoreIndex instance ready for querying.
    """
    from llamaindex_models import get_text_embedding_3_large
    
    # Get controlled embedding model
    embed_model = get_text_embedding_3_large()
    
    # Create index with documents
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True,
    )
    
    # Persist if directory specified
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
    
    return index


def load_vector_index(persist_dir: str) -> VectorStoreIndex:
    """Load a persisted VectorStoreIndex from disk.
    
    Args:
        persist_dir: Directory containing persisted index.
        
    Returns:
        Loaded VectorStoreIndex instance.
    """
    from llamaindex_models import get_text_embedding_3_large
    
    # Recreate storage context
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    
    # Load index with controlled embedding model
    embed_model = get_text_embedding_3_large()
    index = load_index_from_storage(
        storage_context,
        embed_model=embed_model,
    )
    
    return index


def vector_search(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Perform similarity search against the vector index.
    
    Args:
        index: VectorStoreIndex to search.
        query: Query text to search for.
        top_k: Number of top results to return.
        
    Returns:
        List of result dictionaries with text and metadata.
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    
    output = []
    for node in results:
        output.append({
            "text": node.get_content(),
            "score": node.score if hasattr(node, 'score') else None,
            "metadata": node.metadata if hasattr(node, 'metadata') else {},
        })
    
    return output


def get_index_stats(index: VectorStoreIndex) -> Dict[str, Any]:
    """Get statistics about the vector index.
    
    Args:
        index: VectorStoreIndex to analyze.
        
    Returns:
        Dictionary with index statistics.
    """
    return {
        "index_type": "VectorStoreIndex",
        "document_count": len(index.docstore.docs) if hasattr(index.docstore, 'docs') else 0,
        "has_retriever": hasattr(index, 'as_retriever'),
    }
