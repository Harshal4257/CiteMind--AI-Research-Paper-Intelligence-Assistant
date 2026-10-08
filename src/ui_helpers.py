"""Pure presentation helpers shared by the Streamlit UI and unit tests."""

from __future__ import annotations

import re

from src.rag_pipeline import RAGResult


def user_facing_error(result: RAGResult) -> str | None:
    """Convert backend errors into safe, useful UI messages without tracebacks."""
    if not result.error:
        return None
    if result.error == "No sufficiently relevant evidence was retrieved.":
        return "I could not find enough supporting evidence in the indexed papers to answer this confidently."
    if "Question cannot be empty" in result.error:
        return "Please enter a question before submitting."
    if "exceeds" in result.error:
        return result.error
    return "The research assistant could not complete this request. Check the application logs and try again."


def source_rows(result: RAGResult) -> list[dict[str, str | float]]:
    """Create source-table rows from backend metadata without inventing citations."""
    rows: list[dict[str, str | float]] = []
    seen: set[tuple[str, int]] = set()
    for chunk in result.retrieved_chunks:
        metadata = chunk.document.metadata
        key = (str(metadata["filename"]), int(metadata["page_number"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "paper": key[0],
                "page": key[1],
                "relevance_score": round(chunk.relevance_score, 3),
            }
        )
    return rows


def clean_paper_name(filename: str) -> str:
    """Turn a stored filename into a clean but non-invented display label."""
    stem = filename.rsplit(".", maxsplit=1)[0]
    return re.sub(r"[_-]+", " ", stem).strip()


def evidence_strength(result: RAGResult) -> tuple[str, str]:
    """Describe retrieval evidence only; this is not an LLM-confidence estimate."""
    if not result.retrieved_chunks:
        return "Unavailable", "No relevant chunks were retrieved."
    mean_score = sum(chunk.relevance_score for chunk in result.retrieved_chunks) / len(
        result.retrieved_chunks
    )
    if mean_score >= 0.35:
        return "Strong", "Multiple retrieved passages have strong semantic relevance."
    if mean_score >= 0.20:
        return "Moderate", "Retrieved passages provide useful but not perfect semantic support."
    return "Weak", "The answer is based on lower-relevance retrieved evidence."


def evidence_rows(result: RAGResult, excerpt_chars: int = 280) -> list[dict[str, str | float]]:
    """Build compact provenance cards for the collapsed retrieval-transparency panel."""
    rows: list[dict[str, str | float]] = []
    for chunk in result.retrieved_chunks:
        metadata = chunk.document.metadata
        excerpt = " ".join(chunk.document.page_content.split())
        if len(excerpt) > excerpt_chars:
            excerpt = f"{excerpt[:excerpt_chars].rstrip()}..."
        rows.append(
            {
                "paper": clean_paper_name(str(metadata["filename"])),
                "page": int(metadata["page_number"]),
                "chunk_id": str(metadata["chunk_id"]),
                "relevance_score": round(chunk.relevance_score, 3),
                "excerpt": excerpt,
            }
        )
    return rows


def latency_rows(result: RAGResult) -> list[dict[str, str | float]]:
    """Return predictable stage-latency values for compact display."""
    return [
        {"stage": stage.title(), "seconds": result.latency_seconds.get(stage, 0.0)}
        for stage in ("retrieval", "reranking", "generation", "total")
    ]

