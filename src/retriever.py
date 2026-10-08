"""Metadata-aware semantic retrieval for the persisted FAISS index."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """A retrieved document and its vector-store relevance score."""

    document: Document
    relevance_score: float


def retrieve(
    vector_store: FAISS, question: str, settings: Settings
) -> list[RetrievedChunk]:
    """Run notebook-equivalent similarity search and apply its optional threshold."""
    search_k = (
        settings.initial_retrieval_k
        if settings.reranking_enabled
        else settings.retrieval_top_k
    )
    scored_documents = vector_store.similarity_search_with_relevance_scores(
        question, k=search_k
    )

    results: list[RetrievedChunk] = []
    for document, score in scored_documents:
        score = float(score)
        if (
            settings.min_relevance_score is not None
            and score < settings.min_relevance_score
        ):
            continue
        metadata = document.metadata.copy()
        metadata["relevance_score"] = score
        results.append(
            RetrievedChunk(
                document=Document(page_content=document.page_content, metadata=metadata),
                relevance_score=score,
            )
        )

    logger.info("Retrieved %s relevant chunk(s) for a %s-character question", len(results), len(question))
    return results

