from dataclasses import dataclass

from langchain_core.documents import Document

from src.prompts import SAFE_FALLBACK
from src.rag_pipeline import ResearchPaperRAG, unique_sources
from src.retriever import RetrievedChunk


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    def invoke(self, messages):
        assert messages
        return FakeResponse("Transformers use self-attention to enable parallel sequence processing.")


class FakeVectorStore:
    def similarity_search_with_relevance_scores(self, question: str, k: int):
        return [
            (
                Document(
                    page_content="Self-attention reduces sequential computation.",
                    metadata={"filename": "Attention is all you need.pdf", "page_number": 6, "chunk_id": "transformer::p6::c1"},
                ),
                0.9,
            )
        ]


def test_pipeline_returns_answer_sources_and_latency(settings) -> None:
    rag = ResearchPaperRAG(settings, FakeVectorStore(), FakeLLM())
    result = rag.ask("Why use Transformers?")

    assert "parallel" in result.answer
    assert result.sources == ["Attention is all you need.pdf \N{EM DASH} page 6"]
    assert result.error is None
    assert set(result.latency_seconds) == {"retrieval", "reranking", "generation", "total"}


def test_pipeline_returns_safe_fallback_when_evidence_is_below_threshold(settings) -> None:
    class WeakStore:
        def similarity_search_with_relevance_scores(self, question: str, k: int):
            return [(Document(page_content="weak", metadata={"filename": "x.pdf", "page_number": 1, "chunk_id": "x"}), 0.1)]

    rag = ResearchPaperRAG(settings, WeakStore(), FakeLLM())
    result = rag.ask("Unsupported question")

    assert result.answer == SAFE_FALLBACK
    assert result.error == "No sufficiently relevant evidence was retrieved."
    assert result.latency_seconds["generation"] == 0.0


def test_unique_sources_deduplicates_repeated_pages() -> None:
    document = Document(page_content="text", metadata={"filename": "paper.pdf", "page_number": 2, "chunk_id": "a"})
    chunks = [RetrievedChunk(document, 0.9), RetrievedChunk(document, 0.8)]
    assert unique_sources(chunks) == ["paper.pdf \N{EM DASH} page 2"]



