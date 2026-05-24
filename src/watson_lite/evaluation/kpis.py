from __future__ import annotations

import re
import statistics
import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from watson_lite.core.fallbacks import is_fallback_answer_text

if TYPE_CHECKING:
    from watson_lite.core.models import FinalAnswer

_P50_INDEX = 49
_P95_INDEX = 94


@dataclass
class BenchmarkLabel:
    answers: list[str]
    evidence_passages: list[str] = field(default_factory=list)


@dataclass
class KPIReport:
    total_questions: int
    answer_success_rate: float
    grounded_answer_rate: float
    graph_corroboration_rate: float
    type_match_rate: float
    failure_empty_result_rate: float
    latency_p50_s: float
    latency_p95_s: float
    stage_latency_mean_s: dict[str, float]
    cache_hit_rate: float
    accuracy_at_1: float | None
    exact_match: float | None
    f1: float | None
    confidence_calibration_ece: float | None
    retrieval_recall_at_k: float | None
    average_passages_fetched: float
    average_passages_reranked: float
    average_passages_extracted: float


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    no_punc = lowered.translate(str.maketrans("", "", string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    overlap = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(reference))


def _is_success(answer: FinalAnswer) -> bool:
    return answer.confidence > 0.0 and not is_fallback_answer_text(answer.answer)


def _is_failure(answer: FinalAnswer) -> bool:
    if is_fallback_answer_text(answer.answer) or answer.confidence <= 0.0:
        return True
    diagnostics = answer.diagnostics
    if diagnostics is None:
        return False
    return diagnostics.retrieval_empty or diagnostics.extraction_errors > 0


def _average_passage_metric(
    answers: list[FinalAnswer],
    metric_name: str,
) -> float:
    values = [
        float(getattr(answer.diagnostics, metric_name))
        for answer in answers
        if answer.diagnostics is not None
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ece(scores: list[float], labels: list[bool], bins: int = 10) -> float:
    if not scores:
        return 0.0
    total = len(scores)
    ece = 0.0
    for bucket in range(bins):
        lo = bucket / bins
        hi = (bucket + 1) / bins
        indices = [
            i
            for i, score in enumerate(scores)
            if (lo <= score < hi) or (bucket == bins - 1 and score == 1.0)
        ]
        if not indices:
            continue
        conf = sum(scores[i] for i in indices) / len(indices)
        acc = sum(1.0 for i in indices if labels[i]) / len(indices)
        ece += (len(indices) / total) * abs(acc - conf)
    return ece


def _recall_hit(retrieved: list[str], evidence_passages: list[str], top_k: int) -> bool:
    if not evidence_passages:
        return False
    normalized_retrieved = [_normalize_text(text) for text in retrieved[:top_k]]
    normalized_evidence = [_normalize_text(text) for text in evidence_passages]
    for gold in normalized_evidence:
        for candidate in normalized_retrieved:
            if gold and (gold in candidate or candidate in gold):
                return True
    return False


def _latency_percentiles(answers: list[FinalAnswer]) -> tuple[float, float]:
    latencies = [
        float(a.diagnostics.total_latency_s)
        for a in answers
        if a.diagnostics is not None and a.diagnostics.total_latency_s >= 0
    ]
    if not latencies:
        return 0.0, 0.0
    if len(latencies) == 1:
        return latencies[0], latencies[0]
    p50 = statistics.quantiles(latencies, n=100, method="inclusive")[_P50_INDEX]
    p95 = statistics.quantiles(latencies, n=100, method="inclusive")[_P95_INDEX]
    return p50, p95


def _evaluate_labeled(
    answers: list[FinalAnswer],
    labels: list[BenchmarkLabel],
    recall_k: int,
    calibration_bins: int,
) -> tuple[float, float, float, float, float]:
    em_values: list[float] = []
    f1_values: list[float] = []
    correctness: list[bool] = []
    confidence_scores: list[float] = []
    recall_hits = 0
    recall_total = 0

    for answer, label in zip(answers, labels):
        em = max(_exact_match(answer.answer, ref) for ref in label.answers)
        f1 = max(_token_f1(answer.answer, ref) for ref in label.answers)
        em_values.append(em)
        f1_values.append(f1)
        correctness.append(em > 0)
        confidence_scores.append(float(answer.confidence))

        diagnostics = answer.diagnostics
        if diagnostics is not None and label.evidence_passages:
            recall_total += 1
            if _recall_hit(
                diagnostics.top_retrieved_passages,
                label.evidence_passages,
                top_k=recall_k,
            ):
                recall_hits += 1

    total = len(answers)
    accuracy_at_1 = sum(1.0 for ok in correctness if ok) / total
    em_score = sum(em_values) / total
    f1_score = sum(f1_values) / total
    ece = _ece(confidence_scores, correctness, bins=calibration_bins)
    recall = (recall_hits / recall_total) if recall_total else 0.0
    return accuracy_at_1, em_score, f1_score, ece, recall


def evaluate_kpis(
    answers: list[FinalAnswer],
    labels: list[BenchmarkLabel] | None = None,
    *,
    recall_k: int = 10,
    calibration_bins: int = 10,
) -> KPIReport:
    if not answers:
        raise ValueError("answers must not be empty")
    if labels is not None and len(labels) != len(answers):
        raise ValueError("labels length must match answers length")

    total = len(answers)
    success = sum(1 for a in answers if _is_success(a))
    grounded = sum(
        1
        for a in answers
        if a.url.startswith(("http://", "https://")) and len(a.supporting_passages) > 0
    )
    graph = sum(
        1
        for a in answers
        if float(a.confidence_breakdown.get("graph_corroboration", 0)) > 0
    )
    type_match = sum(
        1 for a in answers if float(a.confidence_breakdown.get("type_coercion", 0)) > 0
    )
    failures = sum(1 for a in answers if _is_failure(a))

    latency_p50, latency_p95 = _latency_percentiles(answers)

    stage_series: dict[str, list[float]] = {}
    for answer in answers:
        if answer.diagnostics is None:
            continue
        for stage, value in answer.diagnostics.stage_latencies_s.items():
            stage_series.setdefault(stage, []).append(float(value))
    stage_mean = {
        stage: (sum(values) / len(values))
        for stage, values in stage_series.items()
        if values
    }

    cache_hits = sum(
        a.diagnostics.cache_hits for a in answers if a.diagnostics is not None
    )
    cache_misses = sum(
        a.diagnostics.cache_misses for a in answers if a.diagnostics is not None
    )
    cache_total = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / cache_total) if cache_total else 0.0

    avg_fetched = _average_passage_metric(answers, "passages_fetched")
    avg_reranked = _average_passage_metric(answers, "passages_reranked")
    avg_extracted = _average_passage_metric(answers, "passages_extracted")

    accuracy_at_1: float | None = None
    em_score: float | None = None
    f1_score: float | None = None
    ece: float | None = None
    recall: float | None = None

    if labels is not None:
        accuracy_at_1, em_score, f1_score, ece, recall = _evaluate_labeled(
            answers, labels, recall_k=recall_k, calibration_bins=calibration_bins
        )

    return KPIReport(
        total_questions=total,
        answer_success_rate=success / total,
        grounded_answer_rate=grounded / total,
        graph_corroboration_rate=graph / total,
        type_match_rate=type_match / total,
        failure_empty_result_rate=failures / total,
        latency_p50_s=latency_p50,
        latency_p95_s=latency_p95,
        stage_latency_mean_s=stage_mean,
        cache_hit_rate=cache_hit_rate,
        accuracy_at_1=accuracy_at_1,
        exact_match=em_score,
        f1=f1_score,
        confidence_calibration_ece=ece,
        retrieval_recall_at_k=recall,
        average_passages_fetched=avg_fetched,
        average_passages_reranked=avg_reranked,
        average_passages_extracted=avg_extracted,
    )
