from src.evaluation import EvaluationResult, summarize_results


def test_summarize_results_reports_repeatable_smoke_metrics() -> None:
    summary = summarize_results(
        [
            EvaluationResult("one", True, 1.0, True, None),
            EvaluationResult("two", False, 0.5, True, None),
        ]
    )

    assert summary.case_count == 2
    assert summary.source_hit_rate == 0.5
    assert summary.average_keyword_coverage == 0.75
    assert summary.answer_generation_rate == 1.0
