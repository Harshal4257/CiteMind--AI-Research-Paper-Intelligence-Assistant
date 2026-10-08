"""Thin application service used by the Streamlit interface."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from src.config import Settings
from src.rag_pipeline import RAGResult, ResearchPaperRAG
from src.vectorstore import index_exists

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Cheap application health information; never calls the LLM."""

    papers: tuple[str, ...]
    index_available: bool
    llm_provider: str
    llm_model: str

    @property
    def ready(self) -> bool:
        return bool(self.papers) and self.index_available


def get_health_status(settings: Settings) -> HealthStatus:
    """Inspect document and vector-store availability without model initialization."""
    papers = tuple(sorted(path.name for path in settings.papers_dir.glob("*.pdf")))
    return HealthStatus(
        papers=papers,
        index_available=index_exists(settings.vector_store_dir),
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )


class RAGApplicationService:
    """Stable UI boundary around the existing backend orchestration class."""

    def __init__(self, rag: ResearchPaperRAG) -> None:
        self.rag = rag

    @classmethod
    def create(cls, settings: Settings) -> "RAGApplicationService":
        """Initialize the existing backend once for the Streamlit process."""
        return cls(ResearchPaperRAG.from_settings(settings))

    def ask(self, question: str, answer_depth: str = "Standard") -> RAGResult:
        """Log only request metadata, then delegate to the existing RAG backend."""
        request_id = uuid.uuid4().hex[:8]
        logger.info("RAG request received | request_id=%s | chars=%s", request_id, len(question))
        result = self.rag.ask(question, answer_depth=answer_depth)
        logger.info(
            "RAG request finished | request_id=%s | chunks=%s | total_seconds=%s | error=%s",
            request_id,
            len(result.retrieved_chunks),
            result.latency_seconds.get("total", 0.0),
            bool(result.error),
        )
        return result

