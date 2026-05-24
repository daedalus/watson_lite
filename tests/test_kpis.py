from watson_lite.core.models import AnswerDiagnostics, FinalAnswer
from watson_lite.evaluation.kpis import BenchmarkLabel, evaluate_kpis
import pytest


def _answer(
    text: str,
    confidence: float,
    *,
    graph: float = 0.0,
    type_signal: float = 0.0,
    diagnostics: AnswerDiagnostics | None = None,
    passages: list[str] | None = None,
) -> FinalAnswer:
    return FinalAnswer(
        answer=text,
        confidence=confidence,
        source="src",
        url="https://example.org",
        supporting_passages=passages or ["support"],
        confidence_breakdown={
            "graph_corroboration": graph,
            "type_coercion": type_signal,
        },
        diagnostics=diagnostics,
    )


class TestKPIEvaluation:
    def test_unlabeled_kpis(self) -> None:
        answers = [
            _answer(
                "Gustave Eiffel",
                0.9,
                graph=0.2,
                type_signal=0.8,
                diagnostics=AnswerDiagnostics(
                    total_latency_s=1.2,
                    stage_latencies_s={"nlp": 0.1, "retrieval": 0.5},
                    passages_fetched=12,
                    passages_reranked=10,
                    passages_extracted=5,
                    cache_hits=3,
                    cache_misses=1,
                ),
            ),
            _answer(
                "No answer found",
                0.0,
                diagnostics=AnswerDiagnostics(
                    total_latency_s=2.0,
                    stage_latencies_s={"nlp": 0.2, "retrieval": 0.7},
                    passages_fetched=0,
                    passages_reranked=0,
                    passages_extracted=0,
                    retrieval_empty=True,
                    cache_hits=1,
                    cache_misses=3,
                ),
                passages=[],
            ),
        ]
        report = evaluate_kpis(answers)
        assert report.total_questions == 2
        assert report.answer_success_rate == 0.5
        assert report.graph_corroboration_rate == 0.5
        assert report.type_match_rate == 0.5
        assert report.failure_empty_result_rate == 0.5
        assert report.cache_hit_rate == 0.5
        assert report.average_passages_fetched == 6.0
        assert report.average_passages_reranked == 5.0
        assert report.average_passages_extracted == 2.5
        assert report.latency_p50_s > 0
        assert report.latency_p95_s > 0
        assert report.stage_latency_mean_s["nlp"] == pytest.approx(0.15)

    def test_labeled_kpis(self) -> None:
        answers = [
            _answer(
                "Gustave Eiffel",
                0.92,
                diagnostics=AnswerDiagnostics(
                    total_latency_s=1.0,
                    top_retrieved_passages=[
                        "The Eiffel Tower was designed by Gustave Eiffel in Paris."
                    ],
                ),
            ),
            _answer(
                "Paris",
                0.75,
                diagnostics=AnswerDiagnostics(
                    total_latency_s=1.1,
                    top_retrieved_passages=["Paris is the capital city of France."],
                ),
            ),
        ]
        labels = [
            BenchmarkLabel(
                answers=["Gustave Eiffel"],
                evidence_passages=["designed by Gustave Eiffel"],
            ),
            BenchmarkLabel(
                answers=["Paris", "Paris, France"],
                evidence_passages=["capital city of France"],
            ),
        ]
        report = evaluate_kpis(answers, labels)
        assert report.accuracy_at_1 == 1.0
        assert report.exact_match == 1.0
        assert report.f1 == 1.0
        assert report.retrieval_recall_at_k == 1.0
        assert report.confidence_calibration_ece is not None

    def test_requires_non_empty_answers(self) -> None:
        try:
            evaluate_kpis([])
        except ValueError as exc:
            assert "must not be empty" in str(exc)
        else:
            raise AssertionError("Expected ValueError for empty answers")
