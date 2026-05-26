import math

import pytest

from watson_lite.core.models import AnswerDiagnostics, FinalAnswer
from watson_lite.evaluation.kpis import (
    BenchmarkLabel,
    _histogram_js_divergence,
    _histogram_kl_divergence,
    evaluate_kpis,
)


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
        assert report.confidence_calibration_kl_divergence is not None
        assert report.confidence_calibration_js_divergence is not None
        assert report.confidence_calibration_kl_divergence >= 0.0
        assert report.confidence_calibration_js_divergence >= 0.0

    def test_labeled_kpis_calibration_divergence_is_zero_when_perfectly_calibrated(
        self,
    ) -> None:
        answers = [
            _answer("alpha", 0.25),
            _answer("beta", 0.25),
            _answer("gamma", 0.25),
            _answer("delta", 0.25),
            _answer("echo", 0.75),
            _answer("foxtrot", 0.75),
            _answer("golf", 0.75),
            _answer("hotel", 0.75),
        ]
        labels = [
            BenchmarkLabel(answers=["wrong-one"]),
            BenchmarkLabel(answers=["wrong-two"]),
            BenchmarkLabel(answers=["wrong-three"]),
            BenchmarkLabel(answers=["delta"]),
            BenchmarkLabel(answers=["echo"]),
            BenchmarkLabel(answers=["foxtrot"]),
            BenchmarkLabel(answers=["golf"]),
            BenchmarkLabel(answers=["wrong-four"]),
        ]

        report = evaluate_kpis(answers, labels, calibration_bins=2)

        assert report.confidence_calibration_kl_divergence == pytest.approx(0.0)
        assert report.confidence_calibration_js_divergence == pytest.approx(0.0)

    def test_requires_non_empty_answers(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            evaluate_kpis([])

    def test_average_passage_metrics_ignore_missing_diagnostics(self) -> None:
        answers = [
            _answer(
                "Gustave Eiffel",
                0.9,
                diagnostics=AnswerDiagnostics(
                    passages_fetched=12,
                    passages_reranked=10,
                    passages_extracted=5,
                ),
            ),
            _answer("Paris", 0.8, diagnostics=None),
        ]

        report = evaluate_kpis(answers)

        assert report.average_passages_fetched == 12.0
        assert report.average_passages_reranked == 10.0
        assert report.average_passages_extracted == 5.0


class TestHistogramDivergence:
    def test_identical_distributions_have_zero_divergence(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert _histogram_kl_divergence(values, values) == pytest.approx(0.0, abs=1e-10)
        assert _histogram_js_divergence(values, values) == pytest.approx(0.0, abs=1e-10)

    def test_completely_separate_distributions(self) -> None:
        p_vals = [0.1, 0.1, 0.1, 0.1, 0.1]
        q_vals = [0.9, 0.9, 0.9, 0.9, 0.9]
        kl = _histogram_kl_divergence(p_vals, q_vals)
        jsd = _histogram_js_divergence(p_vals, q_vals)
        assert kl > 0.0
        assert jsd > 0.0
        assert jsd < kl

    def test_confidence_shift_detected(self) -> None:
        well_calibrated = [0.9, 0.8, 0.85, 0.95, 0.7]
        overconfident = [0.99, 0.98, 0.97, 0.99, 0.96]
        kl = _histogram_kl_divergence(well_calibrated, overconfident)
        assert kl > 0.0

    def test_divergence_symmetric_for_jsd_not_kl(self) -> None:
        p_vals = [0.1] * 200
        q_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 25
        kl_pq = _histogram_kl_divergence(p_vals, q_vals)
        kl_qp = _histogram_kl_divergence(q_vals, p_vals)
        jsd_pq = _histogram_js_divergence(p_vals, q_vals)
        jsd_qp = _histogram_js_divergence(q_vals, p_vals)
        assert abs(kl_pq - kl_qp) > 0.01
        assert jsd_pq == pytest.approx(jsd_qp, abs=1e-10)

    def test_empty_list_returns_finite_value(self) -> None:
        assert _histogram_kl_divergence([], [0.1, 0.2]) >= 0.0
        assert _histogram_js_divergence([], [0.1, 0.2]) >= 0.0

    def test_single_value_distribution(self) -> None:
        assert _histogram_kl_divergence([0.5], [0.5]) == pytest.approx(0.0, abs=1e-10)
