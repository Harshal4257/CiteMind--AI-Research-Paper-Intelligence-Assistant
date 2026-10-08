"""Optional cross-encoder reranking, kept off by default as in the notebook."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from src.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CrossEncoderReranker:
    """Ranks already-retrieved chunks by query-passage relevance."""

    model: Any

    @classmethod
    def create(cls, model_name: str) -> "CrossEncoderReranker":
        """Load a sentence-transformers cross encoder."""
        try:
            from sentence_transformers import CrossEncoder

            return cls(model=CrossEncoder(model_name))
        except Exception as exc:
            raise RuntimeError(f"Could not load reranker model '{model_name}'.") from exc

    def rerank(
        self, question: str, chunks: list[RetrievedChunk], final_k: int
    ) -> list[RetrievedChunk]:
        """Return the highest-ranking passages with reranker scores in metadata."""
        if not chunks:
            return []
        scores = self.model.predict(
            [(question, chunk.document.page_content) for chunk in chunks]
        )
        ranked = sorted(
            zip(chunks, scores), key=lambda item: float(item[1]), reverse=True
        )
        results: list[RetrievedChunk] = []
        for chunk, score in ranked[:final_k]:
            metadata = chunk.document.metadata.copy()
            metadata["reranker_score"] = float(score)
            document = Document(page_content=chunk.document.page_content, metadata=metadata)
            results.append(
                RetrievedChunk(document=document, relevance_score=chunk.relevance_score)
            )
        return results

