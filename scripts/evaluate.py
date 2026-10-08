"""Run the small smoke-test evaluation against the persisted index."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings
from src.evaluation import evaluate_cases, summarize_results
from src.logging_config import configure_logging
from src.rag_pipeline import ResearchPaperRAG


def main() -> int:
    settings = Settings.from_env(PROJECT_ROOT)
    configure_logging(settings.log_level)
    try:
        rag = ResearchPaperRAG.from_settings(settings)
        results = evaluate_cases(rag)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print("RAG evaluation summary")
    for result in results:
        print(f"\nQuestion: {result.question}")
        print(f"Expected source retrieved: {result.correct_source_retrieved}")
        print(f"Expected keyword coverage: {result.expected_keyword_coverage:.0%}")
        print(f"Answer generated: {result.answer_generated}")
        if result.error:
            print(f"Error: {result.error}")
    summary = summarize_results(results)
    print("\nAggregate smoke checks")
    print(f"Cases: {summary.case_count}")
    print(f"Expected-source hit rate: {summary.source_hit_rate:.0%}")
    print(f"Average expected-keyword coverage: {summary.average_keyword_coverage:.0%}")
    print(f"Answer-generation rate: {summary.answer_generation_rate:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

