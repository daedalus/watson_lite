from __future__ import annotations

import math
import re
import statistics
import string
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from watson_lite.core.fallbacks import is_fallback_answer_text

if TYPE_CHECKING:
    from watson_lite.core.models import FinalAnswer

_P50_INDEX = 49
_P95_INDEX = 94

NormalizeFn = Callable[[str], str]


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
    confidence_calibration_kl_divergence: float | None
    confidence_calibration_js_divergence: float | None
    retrieval_recall_at_k: float | None
    average_passages_fetched: float
    average_passages_reranked: float
    average_passages_extracted: float


def normalize_squad(text: str) -> str:
    lowered = text.lower()
    no_punc = lowered.translate(str.maketrans("", "", string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def normalize_nq_open(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    lowered = decomposed.lower()
    no_punc = lowered.translate(str.maketrans("", "", string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def normalize_triviaqa(text: str) -> str:
    lowered = text.lower()
    exclude = set(string.punctuation + "\u2018\u2019\u00b4\u0060")
    no_punc = "".join(ch if ch not in exclude else " " for ch in lowered)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    no_underscore = no_articles.replace("_", " ")
    return " ".join(no_underscore.split()).strip()


_normalize_text = normalize_squad


def token_f1(
    prediction: str,
    reference: str,
    normalize_fn: NormalizeFn = normalize_squad,
) -> float:
    pred_tokens = normalize_fn(prediction).split()
    ref_tokens = normalize_fn(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(
    prediction: str,
    reference: str,
    normalize_fn: NormalizeFn = normalize_squad,
) -> float:
    return float(normalize_fn(prediction) == normalize_fn(reference))


def _is_success(answer: FinalAnswer) -> bool:
    return answer.confidence > 0.0 and not is_fallback_answer_text(answer.answer)


def _is_failure(answer: FinalAnswer) -> bool:
    if is_fallback_answer_text(answer.answer) or answer.confidence <= 0.0:
        return True
    diagnostics = answer.diagnostics
    if diagnostics is None:
        return False
    return bool(diagnostics.retrieval_empty) or diagnostics.extraction_errors > 0


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


def _clip_probability(value: float, epsilon: float = 1e-12) -> float:
    return min(max(value, epsilon), 1.0 - epsilon)


def _bernoulli_kl_divergence(
    observed: float,
    predicted: float,
    epsilon: float = 1e-12,
) -> float:
    observed = _clip_probability(observed, epsilon)
    predicted = _clip_probability(predicted, epsilon)
    return observed * math.log(observed / predicted) + (1.0 - observed) * math.log(
        (1.0 - observed) / (1.0 - predicted)
    )


def _bernoulli_js_divergence(
    observed: float,
    predicted: float,
    epsilon: float = 1e-12,
) -> float:
    midpoint = (observed + predicted) / 2.0
    return 0.5 * _bernoulli_kl_divergence(
        observed,
        midpoint,
        epsilon,
    ) + 0.5 * _bernoulli_kl_divergence(predicted, midpoint, epsilon)


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
    normalize_fn: NormalizeFn = normalize_squad,
) -> tuple[float, float, float, float, float, float, float]:
    em_values: list[float] = []
    f1_values: list[float] = []
    correctness: list[bool] = []
    confidence_scores: list[float] = []
    bucket_scores: list[list[float]] = [[] for _ in range(calibration_bins)]
    bucket_correctness: list[list[float]] = [[] for _ in range(calibration_bins)]
    recall_hits = 0
    recall_total = 0

    for answer, label in zip(answers, labels):
        em = max(exact_match(answer.answer, ref, normalize_fn) for ref in label.answers)
        f1 = max(token_f1(answer.answer, ref, normalize_fn) for ref in label.answers)
        em_values.append(em)
        f1_values.append(f1)
        correctness.append(em > 0)
        confidence = float(answer.confidence)
        confidence_scores.append(confidence)
        bucket = min(int(confidence * calibration_bins), calibration_bins - 1)
        bucket_scores[bucket].append(confidence)
        bucket_correctness[bucket].append(float(em > 0))

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
    kl_divergence = 0.0
    js_divergence = 0.0
    for scores, labels_in_bucket in zip(bucket_scores, bucket_correctness):
        if not scores:
            continue
        weight = len(scores) / total
        mean_confidence = sum(scores) / len(scores)
        empirical_accuracy = sum(labels_in_bucket) / len(labels_in_bucket)
        kl_divergence += weight * _bernoulli_kl_divergence(
            empirical_accuracy,
            mean_confidence,
        )
        js_divergence += weight * _bernoulli_js_divergence(
            empirical_accuracy,
            mean_confidence,
        )
    recall = (recall_hits / recall_total) if recall_total else 0.0
    return (
        accuracy_at_1,
        em_score,
        f1_score,
        ece,
        kl_divergence,
        js_divergence,
        recall,
    )


def evaluate_kpis(
    answers: list[FinalAnswer],
    labels: list[BenchmarkLabel] | None = None,
    *,
    recall_k: int = 10,
    calibration_bins: int = 10,
    normalize_fn: NormalizeFn = normalize_squad,
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
    kl_divergence: float | None = None
    js_divergence: float | None = None
    recall: float | None = None

    if labels is not None:
        (
            accuracy_at_1,
            em_score,
            f1_score,
            ece,
            kl_divergence,
            js_divergence,
            recall,
        ) = _evaluate_labeled(
            answers,
            labels,
            recall_k=recall_k,
            calibration_bins=calibration_bins,
            normalize_fn=normalize_fn,
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
        confidence_calibration_kl_divergence=kl_divergence,
        confidence_calibration_js_divergence=js_divergence,
        retrieval_recall_at_k=recall,
        average_passages_fetched=avg_fetched,
        average_passages_reranked=avg_reranked,
        average_passages_extracted=avg_extracted,
    )
