"""Small, transparent RAG smoke-test evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from src.prompts import SAFE_FALLBACK
from src.rag_pipeline import RAGResult, ResearchPaperRAG


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    question: str
    expected_source: str
    expected_keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    question: str
    correct_source_retrieved: bool
    expected_keyword_coverage: float
    answer_generated: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Transparent aggregate for the repository's small repeatable smoke benchmark."""

    case_count: int
    source_hit_rate: float
    average_keyword_coverage: float
    answer_generation_rate: float


DEFAULT_TEST_CASES = (
    EvaluationCase(
        question="What limitation of recurrent models motivated Transformers?",
        expected_source="Attention is all you need.pdf",
        expected_keywords=("recurrent", "parallel"),
    ),
    EvaluationCase(
        question="What does BERT pre-train with?",
        expected_source="BERT.pdf",
        expected_keywords=("masked", "language"),
    ),
    EvaluationCase(
        question="How does dense passage retrieval retrieve evidence?",
        expected_source="Dense Passage Retrieval.pdf",
        expected_keywords=("dense", "passage"),
    ),
)


def evaluate_cases(
    rag: ResearchPaperRAG, cases: tuple[EvaluationCase, ...] = DEFAULT_TEST_CASES
) -> list[EvaluationResult]:
    """Measure only the notebook's implemented source/keyword smoke checks."""
    results: list[EvaluationResult] = []
    for case in cases:
        response: RAGResult = rag.ask(case.question)
        filenames = {
            chunk.document.metadata["filename"] for chunk in response.retrieved_chunks
        }
        answer_lower = response.answer.lower()
        expected_keywords = tuple(keyword.lower() for keyword in case.expected_keywords)
        keyword_coverage = sum(
            keyword in answer_lower for keyword in expected_keywords
        ) / max(1, len(expected_keywords))
        results.append(
            EvaluationResult(
                question=case.question,
                correct_source_retrieved=case.expected_source in filenames,
                expected_keyword_coverage=keyword_coverage,
                answer_generated=bool(response.answer)
                and response.answer != SAFE_FALLBACK,
                error=response.error,
            )
        )
    return results


def summarize_results(results: list[EvaluationResult]) -> EvaluationSummary:
    """Aggregate source, keyword, and generation checks without claiming faithfulness scores."""
    count = len(results)
    if not count:
        return EvaluationSummary(0, 0.0, 0.0, 0.0)
    return EvaluationSummary(
        case_count=count,
        source_hit_rate=sum(result.correct_source_retrieved for result in results) / count,
        average_keyword_coverage=sum(result.expected_keyword_coverage for result in results)
        / count,
        answer_generation_rate=sum(result.answer_generated for result in results) / count,
    )

