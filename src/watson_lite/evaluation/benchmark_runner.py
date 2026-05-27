from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from watson_lite.core.cache import SENTINEL, Cache
from watson_lite.core.config import OPTIONAL_FEATURES, FeatureConfig
from watson_lite.core.models import AnswerDiagnostics, FinalAnswer
from watson_lite.evaluation.kpis import (
    BenchmarkLabel,
    KPIReport,
    NormalizeFn,
    _histogram_js_divergence,
    _histogram_kl_divergence,
    evaluate_kpis,
    exact_match,
    normalize_nq_open,
    normalize_squad,
    normalize_triviaqa,
    token_f1,
)
from watson_lite.pipeline import WatsonLite

_CACHE_DB = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "benchmarks"
    / ".answer_cache.sqlite3"
)
_ANSWER_CACHE: Cache | None = None


def _get_answer_cache() -> Cache:
    global _ANSWER_CACHE
    if _ANSWER_CACHE is None:
        _ANSWER_CACHE = Cache(db_path=str(_CACHE_DB))
    return _ANSWER_CACHE


def _cache_key(profile: str, question: str) -> str:
    h = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return f"benchmark:answer:{profile}:{h}"


def _serialize_answer(answer: FinalAnswer) -> str:
    return json.dumps(asdict(answer), default=str)


def _deserialize_answer(raw: str) -> FinalAnswer:
    d = json.loads(raw)
    diag = d.pop("diagnostics", None)
    if diag is not None:
        d["diagnostics"] = AnswerDiagnostics(**diag)
    return FinalAnswer(**d)


def _normalizer_for_dataset(dataset_path: str) -> NormalizeFn:
    name = Path(dataset_path).stem.lower()
    if "triviaqa" in name:
        return normalize_triviaqa
    if "natural_questions" in name or "nq" in name:
        return normalize_nq_open
    return normalize_squad


@dataclass(frozen=True)
class RegressionThresholds:
    max_accuracy_drop: float = 0.02
    max_exact_match_drop: float = 0.02
    max_f1_drop: float = 0.02
    max_recall_drop: float = 0.02
    metric_tolerance: float = 0.001
    max_latency_p95_s: float | None = None
    max_calibration_kl_increase: float | None = None
    max_calibration_jsd_increase: float | None = None
    max_f1_distribution_kl: float | None = None
    max_f1_distribution_jsd: float | None = None


@dataclass(frozen=True)
class BenchmarkSample:
    question: str
    answers: list[str]
    evidence_passages: list[str]


@dataclass
class BenchmarkProfileResult:
    profile: str
    config: FeatureConfig
    report: KPIReport
    answers_with_metrics: list[dict[str, Any]] = field(default_factory=list)


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    parsed = json.loads(path.read_text(encoding="utf-8"))
    items = parsed.get("samples", []) if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise ValueError("benchmark dataset must be a list or {'samples': [...]}")
    return [item for item in items if isinstance(item, dict)]


def load_benchmark_dataset(path: str) -> list[BenchmarkSample]:
    raw = _load_json_or_jsonl(Path(path))

    samples: list[BenchmarkSample] = []
    for row in raw:
        question = str(row.get("question", "")).strip()
        answers = row.get("answers", [])
        evidence = row.get("evidence_passages", [])
        if not question or not isinstance(answers, list) or not answers:
            continue
        samples.append(
            BenchmarkSample(
                question=question,
                answers=[str(a) for a in answers if str(a).strip()],
                evidence_passages=[str(p) for p in evidence if str(p).strip()],
            )
        )
    if not samples:
        raise ValueError("benchmark dataset is empty or invalid")
    return samples


def build_ablation_profiles() -> dict[str, FeatureConfig]:
    baseline = FeatureConfig.baseline()
    minimal = FeatureConfig.minimal()
    profiles: dict[str, FeatureConfig] = {
        "baseline": baseline,
        "minimal": minimal,
    }
    for feature in OPTIONAL_FEATURES:
        profiles[f"baseline_{feature}_off"] = baseline.with_feature(feature, False)
        profiles[f"minimal_{feature}_on"] = minimal.with_feature(feature, True)
    return profiles


def _run_profile(
    profile: str,
    config: FeatureConfig,
    samples: list[BenchmarkSample],
    *,
    recall_k: int,
    calibration_bins: int,
    normalize_fn: NormalizeFn = normalize_squad,
) -> BenchmarkProfileResult:
    watson = WatsonLite(config=config)
    cache = _get_answer_cache()
    answers: list[FinalAnswer] = []
    labels: list[BenchmarkLabel] = []
    per_answer: list[dict[str, Any]] = []
    total = len(samples)
    for i, sample in enumerate(samples):
        key = _cache_key(profile, sample.question)
        cached = cache.get_or_sentinel(key)
        if cached is not SENTINEL:
            ans = _deserialize_answer(cached)
            tag = "cache/HIT"
        else:
            ans = watson.answer(sample.question, verbose=False)
            cache.set(key, _serialize_answer(ans))
            tag = "cache/MISS"

        answers.append(ans)
        labels.append(
            BenchmarkLabel(
                answers=sample.answers,
                evidence_passages=sample.evidence_passages,
            )
        )

        em = max(exact_match(ans.answer, ref, normalize_fn) for ref in sample.answers)
        f1 = max(token_f1(ans.answer, ref, normalize_fn) for ref in sample.answers)
        latency = (
            ans.diagnostics.total_latency_s if ans.diagnostics is not None else None
        )
        per_answer.append(
            {
                "question": sample.question,
                "generated_answer": ans.answer,
                "reference_answers": sample.answers,
                "exact_match": em,
                "f1": f1,
                "confidence": ans.confidence,
                "latency_s": latency,
            }
        )

        q = sample.question[:80] + ("…" if len(sample.question) > 80 else "")
        a = ans.answer[:60] + ("…" if len(ans.answer) > 60 else "")
        expected = " | ".join(sample.answers[:3])
        if len(sample.answers) > 3:
            expected += f" ( +{len(sample.answers) - 3} more)"
        print(
            f"[{i + 1}/{total}] EM={em:.2f} F1={f1:.2f} conf={ans.confidence:.2f} [{tag}]\n"
            f"  Q: {q}\n  A: {a}\n  ≈: {expected}\n"
        )

    report = evaluate_kpis(
        answers,
        labels,
        recall_k=recall_k,
        calibration_bins=calibration_bins,
        normalize_fn=normalize_fn,
    )
    return BenchmarkProfileResult(
        profile=profile,
        config=config,
        report=report,
        answers_with_metrics=per_answer,
    )


def _metric_drop(
    baseline_value: float | None,
    value: float | None,
) -> float:
    if baseline_value is None or value is None:
        return 0.0
    return baseline_value - value


def _regression_issue(
    profile: str,
    metric: str,
    baseline: float,
    current: float,
    drop: float,
    max_drop: float,
) -> dict[str, float | str]:
    return {
        "profile": profile,
        "metric": metric,
        "baseline": baseline,
        "current": current,
        "drop": drop,
        "max_drop": max_drop,
    }


def _check_distribution_regression(
    issues: list[dict[str, float | str]],
    profile: str,
    baseline_values: list[float],
    current_values: list[float],
    threshold: float | None,
    tolerance: float,
    metric_name: str,
    divergence_fn: Any,  # noqa: ANN401
) -> None:
    if not baseline_values or not current_values or threshold is None:
        return
    div = divergence_fn(baseline_values, current_values)
    if div - threshold > tolerance:
        issues.append(_regression_issue(profile, metric_name, 0.0, div, div, threshold))


def _check_metric_regressions(
    issues: list[dict[str, float | str]],
    profile: str,
    br: Any,  # noqa: ANN401
    cr: Any,  # noqa: ANN401
    metric_checks: tuple[tuple[str, float], ...],
    tolerance: float,
) -> None:
    for metric, max_drop in metric_checks:
        base_value = float(getattr(br, metric, 0.0) or 0.0)
        value = float(getattr(cr, metric, 0.0) or 0.0)
        drop = _metric_drop(base_value, value)
        if drop - max_drop > tolerance:
            issues.append(
                _regression_issue(profile, metric, base_value, value, drop, max_drop)
            )


def _check_calibration_regression(
    issues: list[dict[str, float | str]],
    profile: str,
    max_increase: float | None,
    current_div: float | None,
    baseline_div: float | None,
    tolerance: float,
    metric_name: str,
) -> None:
    if (
        max_increase is not None
        and current_div is not None
        and baseline_div is not None
    ):
        increase = current_div - baseline_div
        if increase - max_increase > tolerance:
            issues.append(
                _regression_issue(
                    profile,
                    metric_name,
                    baseline_div,
                    current_div,
                    increase,
                    max_increase,
                )
            )


def _check_regressions(
    baseline: BenchmarkProfileResult,
    current: BenchmarkProfileResult,
    thresholds: RegressionThresholds,
) -> list[dict[str, float | str]]:
    br = baseline.report
    cr = current.report
    issues: list[dict[str, float | str]] = []
    metric_checks = (
        ("accuracy_at_1", thresholds.max_accuracy_drop),
        ("exact_match", thresholds.max_exact_match_drop),
        ("f1", thresholds.max_f1_drop),
        ("retrieval_recall_at_k", thresholds.max_recall_drop),
    )
    _check_metric_regressions(
        issues, current.profile, br, cr, metric_checks, thresholds.metric_tolerance
    )

    if (
        thresholds.max_latency_p95_s is not None
        and cr.latency_p95_s - thresholds.max_latency_p95_s
        > thresholds.metric_tolerance
    ):
        issues.append(
            _regression_issue(
                current.profile,
                "latency_p95_s",
                br.latency_p95_s,
                cr.latency_p95_s,
                0.0,
                thresholds.max_latency_p95_s,
            )
        )

    _check_calibration_regression(
        issues,
        current.profile,
        thresholds.max_calibration_kl_increase,
        cr.confidence_calibration_kl_divergence,
        br.confidence_calibration_kl_divergence,
        thresholds.metric_tolerance,
        "confidence_calibration_kl_divergence",
    )
    _check_calibration_regression(
        issues,
        current.profile,
        thresholds.max_calibration_jsd_increase,
        cr.confidence_calibration_js_divergence,
        br.confidence_calibration_js_divergence,
        thresholds.metric_tolerance,
        "confidence_calibration_js_divergence",
    )

    baseline_f1 = [a["f1"] for a in baseline.answers_with_metrics]
    current_f1 = [a["f1"] for a in current.answers_with_metrics]
    baseline_conf = [a["confidence"] for a in baseline.answers_with_metrics]
    current_conf = [a["confidence"] for a in current.answers_with_metrics]

    _check_distribution_regression(
        issues,
        current.profile,
        baseline_f1,
        current_f1,
        thresholds.max_f1_distribution_kl,
        thresholds.metric_tolerance,
        "f1_distribution_kl_divergence",
        _histogram_kl_divergence,
    )
    _check_distribution_regression(
        issues,
        current.profile,
        baseline_f1,
        current_f1,
        thresholds.max_f1_distribution_jsd,
        thresholds.metric_tolerance,
        "f1_distribution_js_divergence",
        _histogram_js_divergence,
    )
    _check_distribution_regression(
        issues,
        current.profile,
        baseline_conf,
        current_conf,
        thresholds.max_f1_distribution_kl,
        thresholds.metric_tolerance,
        "confidence_distribution_kl_divergence",
        _histogram_kl_divergence,
    )
    _check_distribution_regression(
        issues,
        current.profile,
        baseline_conf,
        current_conf,
        thresholds.max_f1_distribution_jsd,
        thresholds.metric_tolerance,
        "confidence_distribution_js_divergence",
        _histogram_js_divergence,
    )

    return issues


def _report_row(result: BenchmarkProfileResult) -> dict[str, Any]:
    report = result.report
    return {
        "profile": result.profile,
        "answer_success_rate": report.answer_success_rate,
        "failure_empty_result_rate": report.failure_empty_result_rate,
        "accuracy_at_1": report.accuracy_at_1,
        "exact_match": report.exact_match,
        "f1": report.f1,
        "retrieval_recall_at_k": report.retrieval_recall_at_k,
        "confidence_calibration_ece": report.confidence_calibration_ece,
        "confidence_calibration_kl_divergence": report.confidence_calibration_kl_divergence,
        "confidence_calibration_js_divergence": report.confidence_calibration_js_divergence,
        "latency_p50_s": report.latency_p50_s,
        "latency_p95_s": report.latency_p95_s,
    }


def run_benchmark_profiles(  # pylint: disable=too-many-arguments
    *,
    dataset_path: str,
    config: FeatureConfig,
    output_json_path: str,
    output_csv_path: str | None = None,
    recall_k: int = 10,
    calibration_bins: int = 10,
    ablation_sweep: bool = False,
    regression_check: bool = False,
    thresholds: RegressionThresholds | None = None,
) -> tuple[list[BenchmarkProfileResult], list[dict[str, float | str]]]:
    samples = load_benchmark_dataset(dataset_path)
    normalize_fn = _normalizer_for_dataset(dataset_path)
    profiles = build_ablation_profiles() if ablation_sweep else {"configured": config}
    if regression_check and "baseline" not in profiles:
        profiles = {"baseline": FeatureConfig.baseline(), **profiles}

    results: list[BenchmarkProfileResult] = []
    for name, profile_config in profiles.items():
        results.append(
            _run_profile(
                name,
                profile_config,
                samples,
                recall_k=recall_k,
                calibration_bins=calibration_bins,
                normalize_fn=normalize_fn,
            )
        )

    regressions: list[dict[str, float | str]] = []
    if regression_check:
        baseline_result = next((r for r in results if r.profile == "baseline"), None)
        if baseline_result is None:
            raise ValueError("regression_check requires a baseline profile run")
        active_thresholds = thresholds or RegressionThresholds()
        for result in results:
            if result.profile == "baseline":
                continue
            regressions.extend(
                _check_regressions(
                    baseline=baseline_result,
                    current=result,
                    thresholds=active_thresholds,
                )
            )

    payload = {
        "results": [
            {
                "profile": r.profile,
                "config": asdict(r.config),
                "metrics": asdict(r.report),
                "answers": r.answers_with_metrics,
            }
            for r in results
        ],
        "regressions": regressions,
    }
    Path(output_json_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if output_csv_path:
        rows = [_report_row(r) for r in results]
        with Path(output_csv_path).open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return results, regressions
