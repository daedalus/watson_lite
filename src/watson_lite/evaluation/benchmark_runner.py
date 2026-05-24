from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from watson_lite.core.config import FeatureConfig, OPTIONAL_FEATURES
from watson_lite.evaluation.kpis import BenchmarkLabel, KPIReport, evaluate_kpis
from watson_lite.pipeline import WatsonLite


@dataclass(frozen=True)
class RegressionThresholds:
    max_accuracy_drop: float = 0.02
    max_exact_match_drop: float = 0.02
    max_f1_drop: float = 0.02
    max_recall_drop: float = 0.02
    metric_tolerance: float = 0.001
    max_latency_p95_s: float | None = None


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


def load_benchmark_dataset(path: str) -> list[BenchmarkSample]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    raw: list[dict[str, Any]]
    if suffix == ".jsonl":
        raw = [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        parsed = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            items = parsed.get("samples", [])
        else:
            items = parsed
        if not isinstance(items, list):
            raise ValueError("benchmark dataset must be a list or {'samples': [...]} ")
        raw = [item for item in items if isinstance(item, dict)]

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
) -> BenchmarkProfileResult:
    watson = WatsonLite(config=config)
    answers = [watson.answer(sample.question, verbose=False) for sample in samples]
    labels = [
        BenchmarkLabel(
            answers=sample.answers,
            evidence_passages=sample.evidence_passages,
        )
        for sample in samples
    ]
    report = evaluate_kpis(
        answers,
        labels,
        recall_k=recall_k,
        calibration_bins=calibration_bins,
    )
    return BenchmarkProfileResult(profile=profile, config=config, report=report)


def _metric_drop(
    baseline_value: float | None,
    value: float | None,
) -> float:
    if baseline_value is None or value is None:
        return 0.0
    return baseline_value - value


def _check_regressions(
    baseline: KPIReport,
    current: BenchmarkProfileResult,
    thresholds: RegressionThresholds,
) -> list[dict[str, float | str]]:
    report = current.report
    issues: list[dict[str, float | str]] = []
    checks = (
        ("accuracy_at_1", thresholds.max_accuracy_drop),
        ("exact_match", thresholds.max_exact_match_drop),
        ("f1", thresholds.max_f1_drop),
        ("retrieval_recall_at_k", thresholds.max_recall_drop),
    )
    for metric, max_drop in checks:
        base_value = float(getattr(baseline, metric, 0.0) or 0.0)
        value = float(getattr(report, metric, 0.0) or 0.0)
        drop = _metric_drop(base_value, value)
        if drop - max_drop > thresholds.metric_tolerance:
            issues.append(
                {
                    "profile": current.profile,
                    "metric": metric,
                    "baseline": base_value,
                    "current": value,
                    "drop": drop,
                    "max_drop": max_drop,
                }
            )
    if (
        thresholds.max_latency_p95_s is not None
        and report.latency_p95_s - thresholds.max_latency_p95_s
        > thresholds.metric_tolerance
    ):
        issues.append(
            {
                "profile": current.profile,
                "metric": "latency_p95_s",
                "baseline": baseline.latency_p95_s,
                "current": report.latency_p95_s,
                "drop": 0.0,
                "max_drop": thresholds.max_latency_p95_s,
            }
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
        "latency_p50_s": report.latency_p50_s,
        "latency_p95_s": report.latency_p95_s,
    }


def run_benchmark_profiles(
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
    profiles = (
        build_ablation_profiles()
        if ablation_sweep
        else {"configured": config}
    )
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
                    baseline=baseline_result.report,
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
