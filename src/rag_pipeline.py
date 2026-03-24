"""RAG pipeline helpers for prompt construction, answer generation, and evaluation."""

import re
from typing import Any, Dict, List

from llama_index.core import VectorStoreIndex

from llamaindex_models import get_gpt4o
from vector_db import retrieve_relevant_documents


def build_augmented_prompt(query: str, retrieved_documents: List[Dict[str, Any]]) -> str:
    """Construct an augmented prompt from retrieved documents and a user query."""
    context_sections = []
    for index, document in enumerate(retrieved_documents, start=1):
        metadata = document.get("metadata", {})
        title = metadata.get("title", f"Document {index}")
        score = document.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        context_sections.append(
            f"[{index}] {title}{score_text}\n{document.get('text', '').strip()}"
        )

    context_block = "\n\n".join(context_sections) if context_sections else "No retrieved context available."

    return (
        "You are a retrieval-augmented assistant. Answer the user question using only the provided context. "
        "If the context is insufficient, say that directly.\n\n"
        f"User Question:\n{query}\n\n"
        f"Retrieved Context:\n{context_block}\n\n"
        "Write a concise answer grounded in the retrieved context."
    )


def prepare_prompt_response(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Prepare retrieved documents and an augmented prompt for a query."""
    retrieval_result = retrieve_relevant_documents(index=index, query=query, top_k=top_k)
    prompt = build_augmented_prompt(query=query, retrieved_documents=retrieval_result["documents"])

    return {
        **retrieval_result,
        "prompt": prompt,
    }


def generate_answer_from_prompt(prompt: str) -> str:
    """Send an augmented prompt to GPT-4o and return the synthesized answer."""
    llm = get_gpt4o(temperature=0.1, max_tokens=300)
    response = llm.complete(prompt)
    return response.text if hasattr(response, "text") else str(response)


def answer_user_query(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Run the full query-to-answer flow for a user question."""
    prompt_result = prepare_prompt_response(index=index, query=query, top_k=top_k)
    answer = generate_answer_from_prompt(prompt_result["prompt"])

    return {
        **prompt_result,
        "answer": answer,
    }


def score_generated_answer(expected_answer: str, generated_answer: str) -> float:
    """Compute a lightweight lexical overlap score between expected and generated answers."""
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected_answer.lower()))
    generated_tokens = set(re.findall(r"[a-z0-9]+", generated_answer.lower()))

    if not expected_tokens or not generated_tokens:
        return 0.0

    overlap = expected_tokens.intersection(generated_tokens)
    return round(len(overlap) / len(expected_tokens), 3)


def evaluate_rag_pipeline(
    index: VectorStoreIndex,
    evaluation_examples: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Evaluate the end-to-end RAG answer flow against reference answers."""
    results = []
    for example in evaluation_examples:
        answer_result = answer_user_query(index=index, query=example["query"], top_k=top_k)
        score = score_generated_answer(
            expected_answer=example["expected_answer"],
            generated_answer=answer_result["answer"],
        )
        results.append({
            "query": example["query"],
            "expected_answer": example["expected_answer"],
            "generated_answer": answer_result["answer"],
            "score": score,
        })

    average_score = round(sum(result["score"] for result in results) / len(results), 3) if results else 0.0

    return {
        "example_count": len(results),
        "average_score": average_score,
        "passed_count": sum(1 for result in results if result["score"] >= 0.2),
        "results": results,
    }