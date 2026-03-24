"""Document ingestion pipeline for the RAG system.

Loads documents from various sources (HuggingFace, local files, sample data)
and converts them to LlamaIndex Document format.
"""

from typing import List, Dict, Any
from llama_index.core import Document


def _get_hf_load_dataset():
    """Return HuggingFace load_dataset if available, otherwise None."""
    try:
        from datasets import load_dataset
        return load_dataset
    except Exception:
        return None


def load_sample_documents(count: int = 10) -> List[Document]:
    """Load sample Wikipedia-like documents for development and testing.
    
    Args:
        count: Number of sample documents to generate.
        
    Returns:
        List of LlamaIndex Document objects.
    """
    # Sample document texts (from rag-mini-wikipedia theme)
    sample_texts = [
        "Python is a high-level, interpreted programming language created by Guido van Rossum. First released in 1991, Python's design philosophy emphasizes code readability.",
        "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on tasks.",
        "The Internet is a global system of interconnected computer networks that use the Internet protocol suite to communicate between networks and devices.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
        "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user.",
        "Artificial Intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals and humans.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains. They are fundamental to deep learning models.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "Vector embeddings are numeric representations of text that capture semantic meaning. They are central to modern information retrieval and RAG systems.",
        "Retrieval-augmented generation is a technique that combines information retrieval with generative models to improve the quality of generated text.",
    ]
    
    # Generate documents
    docs = []
    for i in range(min(count, len(sample_texts))):
        doc = Document(
            text=sample_texts[i],
            metadata={
                "source": "sample",
                "index": i,
                "title": f"Sample Document {i}",
            }
        )
        docs.append(doc)
    
    # If more documents are requested than available, cycle through samples
    while len(docs) < count:
        i = len(docs) % len(sample_texts)
        doc = Document(
            text=sample_texts[i],
            metadata={
                "source": "sample",
                "index": len(docs),
                "title": f"Sample Document {len(docs)}",
            }
        )
        docs.append(doc)
    
    return docs


def load_huggingface_documents(count: int = 50) -> List[Document]:
    """Load documents from HuggingFace rag-mini-wikipedia dataset.
    
    This function will load from:
    hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet
    
    Falls back to sample documents if HuggingFace loading is unavailable.

    Args:
        count: Number of documents to load.
    
    Returns:
        List of LlamaIndex Document objects.
    """
    load_dataset = _get_hf_load_dataset()
    if load_dataset is None:
        return load_sample_documents(count=count)

    try:
        dataset = load_dataset(
            "rag-datasets/rag-mini-wikipedia",
            "text-corpus",
            split="passages",
        )

        docs: List[Document] = []
        limit = min(count, len(dataset))
        for i in range(limit):
            row = dataset[i]
            text = str(row.get("passage", "")).strip()
            if not text:
                continue

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source": "huggingface",
                        "dataset": "rag-datasets/rag-mini-wikipedia",
                        "split": "passages",
                        "index": i,
                        "id": row.get("id", i),
                    },
                )
            )

        return docs if docs else load_sample_documents(count=count)
    except Exception:
        return load_sample_documents(count=count)


def load_sample_evaluation_examples(count: int = 5) -> List[Dict[str, Any]]:
    """Load sample question-answer pairs for RAG evaluation.

    Args:
        count: Number of examples to return.

    Returns:
        List of dictionaries containing a query and expected answer.
    """
    examples = [
        {
            "query": "What is Python?",
            "expected_answer": "Python is a high-level, interpreted programming language created by Guido van Rossum.",
            "reference_title": "Sample Document 0",
        },
        {
            "query": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of artificial intelligence focused on algorithms that improve performance on tasks.",
            "reference_title": "Sample Document 1",
        },
        {
            "query": "What is the Internet?",
            "expected_answer": "The Internet is a global system of interconnected computer networks that communicate using the Internet protocol suite.",
            "reference_title": "Sample Document 2",
        },
        {
            "query": "What are vector embeddings?",
            "expected_answer": "Vector embeddings are numeric representations of text that capture semantic meaning.",
            "reference_title": "Sample Document 8",
        },
        {
            "query": "What is retrieval-augmented generation?",
            "expected_answer": "Retrieval-augmented generation combines information retrieval with generative models to improve generated text.",
            "reference_title": "Sample Document 9",
        },
    ]

    return examples[:count]


def load_huggingface_test_questions(count: int = 5) -> List[Dict[str, Any]]:
    """Load evaluation questions for the RAG system.

    This is intended to eventually load from:
    hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet

    Args:
        count: Number of examples to return.

    Returns:
        List of question-answer evaluation examples.
    """
    load_dataset = _get_hf_load_dataset()
    if load_dataset is None:
        return load_sample_evaluation_examples(count=count)

    try:
        dataset = load_dataset(
            "rag-datasets/rag-mini-wikipedia",
            "question-answer",
            split="test",
        )

        examples: List[Dict[str, Any]] = []
        limit = min(count, len(dataset))
        for i in range(limit):
            row = dataset[i]
            query = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not query or not answer:
                continue

            examples.append(
                {
                    "query": query,
                    "expected_answer": answer,
                    "id": row.get("id", i),
                    "source": "huggingface",
                }
            )

        return examples if examples else load_sample_evaluation_examples(count=count)
    except Exception:
        return load_sample_evaluation_examples(count=count)
