"""Vector database operations using LlamaIndex VectorStoreIndex.

Manages creation, persistence, and queries against the local vector database.
"""

import json
import os
import shutil
from typing import Any, List, Dict, Optional, Set
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document, BaseNode, TextNode


def _default_resume_id(document: Document, fallback_index: int) -> str:
    """Derive a stable resume id from metadata so retries can skip processed docs."""
    metadata = document.metadata if isinstance(document.metadata, dict) else {}
    source = str(metadata.get("source", "unknown"))
    dataset_index = metadata.get("dataset_index")
    doc_id = metadata.get("id")
    index_value = metadata.get("index", fallback_index)

    if dataset_index is not None:
        return f"{source}:dataset_index:{dataset_index}"
    if doc_id is not None:
        return f"{source}:id:{doc_id}"
    return f"{source}:index:{index_value}"


def _ensure_resume_ids(documents: List[Document]) -> List[Document]:
    """Attach a stable resume_id in metadata to every document."""
    for i, document in enumerate(documents):
        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        if "resume_id" not in metadata:
            metadata["resume_id"] = _default_resume_id(document, i)
        document.metadata = metadata
    return documents


def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_checkpoint(
    checkpoint_path: str,
    resume_key: str,
    processed_ids: Set[str],
    total_documents: int,
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "resume_key": resume_key,
        "total_documents": total_documents,
        "processed_ids": sorted(processed_ids),
        "processed_count": len(processed_ids),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _clear_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def _persist_progress(
    index: VectorStoreIndex,
    persist_dir: str,
    checkpoint_path: str,
    resume_key: str,
    processed_ids: Set[str],
    total_documents: int,
) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    _save_checkpoint(
        checkpoint_path=checkpoint_path,
        resume_key=resume_key,
        processed_ids=processed_ids,
        total_documents=total_documents,
    )


def create_vector_index(
    documents: List[Document],
    persist_dir: Optional[str] = None,
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
    resume_key: Optional[str] = None,
    batch_size: int = 50,
) -> VectorStoreIndex:
    """Create a VectorStoreIndex from documents using controlled embedding model.
    
    Args:
        documents: List of LlamaIndex Document objects to index.
        persist_dir: Optional directory to persist the index.
        resume: If true, persist incremental progress and resume retries.
        checkpoint_path: Path to a JSON checkpoint file.
        resume_key: Signature of the requested ingest config (source + count).
        batch_size: Persist checkpoint every N inserted documents.
        
    Returns:
        VectorStoreIndex instance ready for querying.
    """
    from llamaindex_models import get_text_embedding_3_large

    if not documents:
        raise ValueError("No documents provided for indexing.")

    embed_model = get_text_embedding_3_large()

    if not resume or not persist_dir or not checkpoint_path or not resume_key:
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            show_progress=True,
        )

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)

        return index

    safe_batch_size = max(1, int(batch_size))
    docs_with_ids = _ensure_resume_ids(documents)

    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_key = checkpoint.get("resume_key")
    key_matches = checkpoint_key == resume_key

    if not key_matches:
        _clear_directory(persist_dir)
        checkpoint = {}

    processed_ids: Set[str] = set()
    if key_matches:
        saved_ids = checkpoint.get("processed_ids", [])
        if isinstance(saved_ids, list):
            processed_ids = {str(item) for item in saved_ids}

    index: Optional[VectorStoreIndex] = None
    if processed_ids and os.path.isdir(persist_dir):
        try:
            index = load_vector_index(persist_dir=persist_dir)
        except Exception:
            index = None
            processed_ids = set()
            _clear_directory(persist_dir)

    remaining_docs = [
        doc for doc in docs_with_ids
        if str(doc.metadata.get("resume_id")) not in processed_ids
    ]

    if index is None:
        first_batch = remaining_docs[:safe_batch_size]
        if not first_batch:
            # Nothing left to process for this key, but index not loadable.
            # Start fresh to ensure a usable index exists.
            first_batch = docs_with_ids[:safe_batch_size]
            processed_ids = set()

        index = VectorStoreIndex.from_documents(
            first_batch,
            embed_model=embed_model,
            show_progress=True,
        )
        processed_ids.update(str(doc.metadata.get("resume_id")) for doc in first_batch)
        _persist_progress(
            index=index,
            persist_dir=persist_dir,
            checkpoint_path=checkpoint_path,
            resume_key=resume_key,
            processed_ids=processed_ids,
            total_documents=len(docs_with_ids),
        )
        remaining_docs = [
            doc for doc in docs_with_ids
            if str(doc.metadata.get("resume_id")) not in processed_ids
        ]

    inserted_since_persist = 0
    for doc in remaining_docs:
        index.insert(doc)
        processed_ids.add(str(doc.metadata.get("resume_id")))
        inserted_since_persist += 1

        if inserted_since_persist >= safe_batch_size:
            _persist_progress(
                index=index,
                persist_dir=persist_dir,
                checkpoint_path=checkpoint_path,
                resume_key=resume_key,
                processed_ids=processed_ids,
                total_documents=len(docs_with_ids),
            )
            inserted_since_persist = 0

    _persist_progress(
        index=index,
        persist_dir=persist_dir,
        checkpoint_path=checkpoint_path,
        resume_key=resume_key,
        processed_ids=processed_ids,
        total_documents=len(docs_with_ids),
    )

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


def embed_query(query: str) -> List[float]:
    """Convert a user query into an embedding vector.

    Args:
        query: Query text to embed.

    Returns:
        Query embedding vector.
    """
    from llamaindex_models import get_text_embedding_3_large

    embed_model = get_text_embedding_3_large()
    return embed_model.get_query_embedding(query)


def retrieve_relevant_documents(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Run the query-to-retrieval flow for a user query.

    Args:
        index: VectorStoreIndex to search.
        query: User query text.
        top_k: Number of top results to return.

    Returns:
        Dictionary containing query metadata and retrieved documents.
    """
    query_embedding = embed_query(query)
    results = vector_search(index=index, query=query, top_k=top_k)

    return {
        "query": query,
        "top_k": top_k,
        "query_embedding_dimension": len(query_embedding),
        "documents": results,
    }


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
