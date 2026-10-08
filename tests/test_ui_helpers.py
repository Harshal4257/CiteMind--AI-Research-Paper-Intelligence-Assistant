from langchain_core.documents import Document

from src.rag_pipeline import RAGResult
from src.retriever import RetrievedChunk
from src.ui_helpers import (
    clean_paper_name,
    evidence_rows,
    evidence_strength,
    latency_rows,
    source_rows,
    user_facing_error,
)


def _result() -> RAGResult:
    document = Document(
        page_content="evidence",
        metadata={"filename": "paper.pdf", "page_number": 4, "chunk_id": "paper::p4::c1"},
    )
    return RAGResult(
        question="question",
        answer="answer",
        retrieved_chunks=[RetrievedChunk(document, 0.8), RetrievedChunk(document, 0.7)],
        latency_seconds={"retrieval": 0.1, "total": 0.2},
    )


def test_source_rows_deduplicate_file_and_page() -> None:
    rows = source_rows(_result())
    assert rows == [{"paper": "paper.pdf", "page": 4, "relevance_score": 0.8}]


def test_safe_fallback_message_is_user_friendly() -> None:
    result = RAGResult(
        question="unknown",
        error="No sufficiently relevant evidence was retrieved.",
    )
    assert "supporting evidence" in user_facing_error(result)


def test_latency_rows_have_all_stages() -> None:
    assert [row["stage"] for row in latency_rows(_result())] == [
        "Retrieval",
        "Reranking",
        "Generation",
        "Total",
    ]


def test_evidence_strength_is_explicitly_retrieval_based() -> None:
    strength, note = evidence_strength(_result())
    assert strength == "Strong"
    assert "retrieved" in note.lower()


def test_evidence_rows_provide_clean_names_and_compact_excerpt() -> None:
    rows = evidence_rows(_result())
    assert rows[0]["paper"] == "paper"
    assert rows[0]["page"] == 4
    assert rows[0]["excerpt"] == "evidence"


def test_clean_paper_name_removes_extension_without_inventing_title() -> None:
    assert clean_paper_name("attention_is-all-you-need.pdf") == "attention is all you need"

