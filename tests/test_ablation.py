"""Ablation tests for the benchmark-runner's profile-building and evaluation logic.

Each test class isolates one concern:

* ``TestBuildAblationProfiles``     – feature-flag correctness of every generated profile
* ``TestAblationSweepExecution``    – WatsonLite receives the right config per profile
* ``TestAblationOutputArtifacts``   – JSON / CSV artefact structure
* ``TestRegressionDetection``       – metric-drop helper and _check_regressions logic
* ``TestAblationDatasetFormats``    – JSONL dataset path exercised end-to-end
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from watson_lite.core.config import OPTIONAL_FEATURES, FeatureConfig
from watson_lite.core.models import AnswerDiagnostics, FinalAnswer
from watson_lite.evaluation.benchmark_runner import (
    BenchmarkProfileResult,
    RegressionThresholds,
    _check_regressions,
    _metric_drop,
    build_ablation_profiles,
    run_benchmark_profiles,
)
from watson_lite.evaluation.kpis import KPIReport

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _final_answer(
    text: str,
    confidence: float,
    *,
    latency: float = 0.1,
    passages: list[str] | None = None,
) -> FinalAnswer:
    return FinalAnswer(
        answer=text,
        confidence=confidence,
        source="src",
        url="https://example.org",
        supporting_passages=["support passage"],
        confidence_breakdown={"graph_corroboration": 0.0, "type_coercion": 0.0},
        diagnostics=AnswerDiagnostics(
            total_latency_s=latency,
            top_retrieved_passages=passages or ["Paris is the capital city of France."],
        ),
    )


def _make_kpi_report(
    accuracy: float = 1.0,
    em: float = 1.0,
    f1: float = 1.0,
    recall: float = 1.0,
    latency_p95: float = 0.5,
) -> KPIReport:
    return KPIReport(
        total_questions=1,
        answer_success_rate=accuracy,
        grounded_answer_rate=1.0,
        graph_corroboration_rate=0.0,
        type_match_rate=0.0,
        failure_empty_result_rate=0.0,
        latency_p50_s=0.1,
        latency_p95_s=latency_p95,
        stage_latency_mean_s={},
        cache_hit_rate=0.0,
        accuracy_at_1=accuracy,
        exact_match=em,
        f1=f1,
        confidence_calibration_ece=0.0,
        retrieval_recall_at_k=recall,
        average_passages_fetched=1.0,
        average_passages_reranked=1.0,
        average_passages_extracted=1.0,
    )


def _write_dataset(
    path: Path, samples: list[dict[str, Any]] | None = None
) -> None:
    if samples is None:
        samples = [
            {
                "question": "What is the capital of France?",
                "answers": ["Paris"],
                "evidence_passages": ["Paris is the capital city of France."],
            }
        ]
    path.write_text(json.dumps(samples), encoding="utf-8")


# ---------------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------------


class TestBuildAblationProfiles:
    def test_baseline_profile_has_all_features_enabled(self) -> None:
        profiles = build_ablation_profiles()
        baseline = profiles["baseline"]
        for feature in OPTIONAL_FEATURES:
            assert getattr(baseline, feature) is True, (
                f"baseline should have {feature}=True"
            )

    def test_minimal_profile_has_all_features_disabled(self) -> None:
        profiles = build_ablation_profiles()
        minimal = profiles["minimal"]
        for feature in OPTIONAL_FEATURES:
            assert getattr(minimal, feature) is False, (
                f"minimal should have {feature}=False"
            )

    def test_baseline_off_profiles_disable_only_the_named_feature(self) -> None:
        profiles = build_ablation_profiles()
        for feature in OPTIONAL_FEATURES:
            profile_name = f"baseline_{feature}_off"
            cfg = profiles[profile_name]
            assert getattr(cfg, feature) is False, (
                f"{profile_name}: expected {feature}=False"
            )
            for other in OPTIONAL_FEATURES:
                if other == feature:
                    continue
                assert getattr(cfg, other) is True, (
                    f"{profile_name}: expected {other}=True"
                )

    def test_minimal_on_profiles_enable_only_the_named_feature(self) -> None:
        profiles = build_ablation_profiles()
        for feature in OPTIONAL_FEATURES:
            profile_name = f"minimal_{feature}_on"
            cfg = profiles[profile_name]
            assert getattr(cfg, feature) is True, (
                f"{profile_name}: expected {feature}=True"
            )
            for other in OPTIONAL_FEATURES:
                if other == feature:
                    continue
                assert getattr(cfg, other) is False, (
                    f"{profile_name}: expected {other}=False"
                )

    def test_each_optional_feature_has_both_variant_profiles(self) -> None:
        profiles = build_ablation_profiles()
        for feature in OPTIONAL_FEATURES:
            assert f"baseline_{feature}_off" in profiles, (
                f"Missing profile baseline_{feature}_off"
            )
            assert f"minimal_{feature}_on" in profiles, (
                f"Missing profile minimal_{feature}_on"
            )

    def test_all_profile_values_are_featureconfig_instances(self) -> None:
        for name, cfg in build_ablation_profiles().items():
            assert isinstance(cfg, FeatureConfig), (
                f"Profile '{name}' is not a FeatureConfig instance"
            )

    def test_baseline_off_profile_differs_from_baseline_by_exactly_one_flag(
        self,
    ) -> None:
        profiles = build_ablation_profiles()
        baseline = profiles["baseline"]
        for feature in OPTIONAL_FEATURES:
            cfg = profiles[f"baseline_{feature}_off"]
            differing = [
                f
                for f in OPTIONAL_FEATURES
                if getattr(cfg, f) != getattr(baseline, f)
            ]
            assert differing == [feature], (
                f"baseline_{feature}_off should differ from baseline only in {feature}"
            )

    def test_minimal_on_profile_differs_from_minimal_by_exactly_one_flag(
        self,
    ) -> None:
        profiles = build_ablation_profiles()
        minimal = profiles["minimal"]
        for feature in OPTIONAL_FEATURES:
            cfg = profiles[f"minimal_{feature}_on"]
            differing = [
                f
                for f in OPTIONAL_FEATURES
                if getattr(cfg, f) != getattr(minimal, f)
            ]
            assert differing == [feature], (
                f"minimal_{feature}_on should differ from minimal only in {feature}"
            )


# ---------------------------------------------------------------------------
# Sweep execution — correct config is forwarded to WatsonLite per profile
# ---------------------------------------------------------------------------


class TestAblationSweepExecution:
    def test_ablation_sweep_instantiates_watson_with_each_profile_config(
        self, tmp_path: Path
    ) -> None:
        dataset = tmp_path / "ds.json"
        output_json = tmp_path / "out.json"
        _write_dataset(dataset)

        received: list[FeatureConfig] = []

        class FakeWatson:
            def __init__(self, config: FeatureConfig) -> None:
                received.append(config)

            def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
                del question, verbose
                return _final_answer("Paris", 0.9)

        with patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson):
            run_benchmark_profiles(
                dataset_path=str(dataset),
                config=FeatureConfig.baseline(),
                output_json_path=str(output_json),
                ablation_sweep=True,
            )

        expected = set(build_ablation_profiles().values())
        assert set(received) == expected

    def test_non_ablation_sweep_uses_only_the_provided_config(
        self, tmp_path: Path
    ) -> None:
        dataset = tmp_path / "ds.json"
        output_json = tmp_path / "out.json"
        _write_dataset(dataset)

        received: list[FeatureConfig] = []
        custom = FeatureConfig.minimal()

        class FakeWatson:
            def __init__(self, config: FeatureConfig) -> None:
                received.append(config)

            def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
                del question, verbose
                return _final_answer("Paris", 0.9)

        with patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson):
            run_benchmark_profiles(
                dataset_path=str(dataset),
                config=custom,
                output_json_path=str(output_json),
            )

        assert received == [custom]

    def test_ablation_sweep_produces_distinct_metrics_per_feature(
        self, tmp_path: Path
    ) -> None:
        """A profile where the answer differs from baseline yields different accuracy."""
        dataset = tmp_path / "ds.json"
        output_json = tmp_path / "out.json"
        _write_dataset(dataset)

        class FakeWatson:
            def __init__(self, config: FeatureConfig) -> None:
                self._cfg = config

            def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
                del question, verbose
                if self._cfg == FeatureConfig.baseline():
                    return _final_answer("Paris", 0.9)
                return _final_answer("Wrong", 0.9)

        with patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson):
            results, _ = run_benchmark_profiles(
                dataset_path=str(dataset),
                config=FeatureConfig.baseline(),
                output_json_path=str(output_json),
                ablation_sweep=True,
                regression_check=True,
            )

        baseline_result = next(r for r in results if r.profile == "baseline")
        other_results = [r for r in results if r.profile != "baseline"]
        assert baseline_result.report.accuracy_at_1 == 1.0
        assert all(r.report.accuracy_at_1 == 0.0 for r in other_results)


# ---------------------------------------------------------------------------
# Output artefacts
# ---------------------------------------------------------------------------


class TestAblationOutputArtifacts:
    def _run_sweep(
        self,
        tmp_path: Path,
        *,
        csv_path: Path | None = None,
    ) -> tuple[list[BenchmarkProfileResult], Path]:
        dataset = tmp_path / "ds.json"
        output_json = tmp_path / "out.json"
        _write_dataset(dataset)

        class FakeWatson:
            def __init__(self, config: FeatureConfig) -> None: ...

            def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
                del question, verbose
                return _final_answer("Paris", 0.9)

        with patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson):
            results, _ = run_benchmark_profiles(
                dataset_path=str(dataset),
                config=FeatureConfig.baseline(),
                output_json_path=str(output_json),
                output_csv_path=str(csv_path) if csv_path else None,
                ablation_sweep=True,
            )
        return results, output_json

    def test_output_json_contains_all_ablation_profile_names(
        self, tmp_path: Path
    ) -> None:
        _, output_json = self._run_sweep(tmp_path)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        found = {r["profile"] for r in payload["results"]}
        for name in build_ablation_profiles():
            assert name in found, f"Profile '{name}' missing from JSON output"

    def test_output_json_config_entries_contain_all_optional_features(
        self, tmp_path: Path
    ) -> None:
        _, output_json = self._run_sweep(tmp_path)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        for entry in payload["results"]:
            assert "config" in entry
            for feature in OPTIONAL_FEATURES:
                assert feature in entry["config"], (
                    f"Feature '{feature}' missing from config of profile "
                    f"'{entry['profile']}'"
                )

    def test_output_json_always_has_regressions_key(self, tmp_path: Path) -> None:
        _, output_json = self._run_sweep(tmp_path)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert "regressions" in payload

    def test_output_csv_row_count_matches_profile_count(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "out.csv"
        results, _ = self._run_sweep(tmp_path, csv_path=csv_path)
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(results)

    def test_output_csv_contains_profile_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "out.csv"
        self._run_sweep(tmp_path, csv_path=csv_path)
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert "profile" in (reader.fieldnames or [])


# ---------------------------------------------------------------------------
# Regression detection helpers
# ---------------------------------------------------------------------------


class TestRegressionDetection:
    # _metric_drop -----------------------------------------------------------

    def test_metric_drop_returns_correct_difference(self) -> None:
        assert _metric_drop(0.9, 0.7) == pytest.approx(0.2)

    def test_metric_drop_returns_zero_when_baseline_is_none(self) -> None:
        assert _metric_drop(None, 0.7) == 0.0

    def test_metric_drop_returns_zero_when_current_is_none(self) -> None:
        assert _metric_drop(0.9, None) == 0.0

    def test_metric_drop_returns_zero_when_both_none(self) -> None:
        assert _metric_drop(None, None) == 0.0

    def test_metric_drop_negative_when_current_exceeds_baseline(self) -> None:
        assert _metric_drop(0.5, 0.8) == pytest.approx(-0.3)

    # _check_regressions — no issues ----------------------------------------

    def test_no_regression_when_metrics_are_within_threshold(self) -> None:
        baseline = _make_kpi_report(accuracy=0.9, em=0.9, f1=0.9, recall=0.9)
        current = BenchmarkProfileResult(
            profile="p",
            config=FeatureConfig.baseline(),
            report=_make_kpi_report(accuracy=0.89, em=0.89, f1=0.89, recall=0.89),
        )
        thresholds = RegressionThresholds(
            max_accuracy_drop=0.02,
            max_exact_match_drop=0.02,
            max_f1_drop=0.02,
            max_recall_drop=0.02,
            metric_tolerance=0.001,
        )
        assert _check_regressions(baseline, current, thresholds) == []

    def test_no_regression_when_metrics_equal_baseline(self) -> None:
        report = _make_kpi_report()
        current = BenchmarkProfileResult(
            profile="p", config=FeatureConfig.baseline(), report=report
        )
        assert _check_regressions(report, current, RegressionThresholds()) == []

    # _check_regressions — metric drops -------------------------------------

    def test_detects_accuracy_drop(self) -> None:
        baseline = _make_kpi_report(accuracy=1.0)
        current = BenchmarkProfileResult(
            profile="bad",
            config=FeatureConfig.minimal(),
            report=_make_kpi_report(accuracy=0.5),
        )
        issues = _check_regressions(
            baseline,
            current,
            RegressionThresholds(max_accuracy_drop=0.0, metric_tolerance=0.0),
        )
        acc_issues = [i for i in issues if i["metric"] == "accuracy_at_1"]
        assert len(acc_issues) == 1
        assert acc_issues[0]["drop"] == pytest.approx(0.5)

    def test_detects_exact_match_drop(self) -> None:
        baseline = _make_kpi_report(em=1.0)
        current = BenchmarkProfileResult(
            profile="p",
            config=FeatureConfig.minimal(),
            report=_make_kpi_report(em=0.0),
        )
        issues = _check_regressions(
            baseline,
            current,
            RegressionThresholds(max_exact_match_drop=0.0, metric_tolerance=0.0),
        )
        assert any(i["metric"] == "exact_match" for i in issues)

    def test_detects_f1_drop(self) -> None:
        baseline = _make_kpi_report(f1=1.0)
        current = BenchmarkProfileResult(
            profile="p",
            config=FeatureConfig.minimal(),
            report=_make_kpi_report(f1=0.0),
        )
        issues = _check_regressions(
            baseline,
            current,
            RegressionThresholds(max_f1_drop=0.0, metric_tolerance=0.0),
        )
        assert any(i["metric"] == "f1" for i in issues)

    def test_detects_recall_drop(self) -> None:
        baseline = _make_kpi_report(recall=1.0)
        current = BenchmarkProfileResult(
            profile="p",
            config=FeatureConfig.minimal(),
            report=_make_kpi_report(recall=0.0),
        )
        issues = _check_regressions(
            baseline,
            current,
            RegressionThresholds(max_recall_drop=0.0, metric_tolerance=0.0),
        )
        assert any(i["metric"] == "retrieval_recall_at_k" for i in issues)

    # _check_regressions — latency ------------------------------------------

    def test_detects_latency_regression(self) -> None:
        baseline = _make_kpi_report(latency_p95=0.5)
        current = BenchmarkProfileResult(
            profile="slow",
            config=FeatureConfig.baseline(),
            report=_make_kpi_report(latency_p95=2.0),
        )
        thresholds = RegressionThresholds(
            max_latency_p95_s=1.0, metric_tolerance=0.0
        )
        issues = _check_regressions(baseline, current, thresholds)
        latency_issues = [i for i in issues if i["metric"] == "latency_p95_s"]
        assert len(latency_issues) == 1

    def test_no_latency_regression_when_threshold_is_none(self) -> None:
        baseline = _make_kpi_report(latency_p95=0.1)
        current = BenchmarkProfileResult(
            profile="p",
            config=FeatureConfig.baseline(),
            report=_make_kpi_report(latency_p95=99.0),
        )
        issues = _check_regressions(
            baseline, current, RegressionThresholds(max_latency_p95_s=None)
        )
        assert not any(i["metric"] == "latency_p95_s" for i in issues)

    def test_no_latency_regression_when_within_threshold(self) -> None:
        baseline = _make_kpi_report(latency_p95=1.0)
        current = BenchmarkProfileResult(
            profile="p",
            config=FeatureConfig.baseline(),
            report=_make_kpi_report(latency_p95=1.5),
        )
        issues = _check_regressions(
            baseline,
            current,
            RegressionThresholds(max_latency_p95_s=2.0, metric_tolerance=0.001),
        )
        assert not any(i["metric"] == "latency_p95_s" for i in issues)

    # regression_check injects baseline -------------------------------------

    def test_regression_check_injects_baseline_profile_when_missing(
        self, tmp_path: Path
    ) -> None:
        """When ablation_sweep=False and regression_check=True, baseline is injected."""
        dataset = tmp_path / "ds.json"
        output_json = tmp_path / "out.json"
        _write_dataset(dataset)

        class FakeWatson:
            def __init__(self, config: FeatureConfig) -> None:
                self._cfg = config

            def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
                del question, verbose
                return _final_answer("Paris", 0.9)

        with patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson):
            results, regressions = run_benchmark_profiles(
                dataset_path=str(dataset),
                config=FeatureConfig.minimal(),
                output_json_path=str(output_json),
                ablation_sweep=False,
                regression_check=True,
            )

        profile_names = [r.profile for r in results]
        assert "baseline" in profile_names
        assert regressions == []

    def test_run_sweep_with_regression_check_returns_regressions_list(
        self, tmp_path: Path
    ) -> None:
        dataset = tmp_path / "ds.json"
        output_json = tmp_path / "out.json"
        _write_dataset(dataset)

        class FakeWatson:
            def __init__(self, config: FeatureConfig) -> None:
                self._cfg = config

            def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
                del question, verbose
                if self._cfg == FeatureConfig.baseline():
                    return _final_answer("Paris", 0.9)
                return _final_answer("Wrong", 0.8, passages=["unrelated text"])

        with patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson):
            _, regressions = run_benchmark_profiles(
                dataset_path=str(dataset),
                config=FeatureConfig.minimal(),
                output_json_path=str(output_json),
                ablation_sweep=False,
                regression_check=True,
                thresholds=RegressionThresholds(
                    max_accuracy_drop=0.0,
                    max_exact_match_drop=0.0,
                    max_f1_drop=0.0,
                    max_recall_drop=0.0,
                    metric_tolerance=0.0,
                ),
            )

        assert regressions


# ---------------------------------------------------------------------------
# Dataset format variants
# ---------------------------------------------------------------------------


class TestAblationDatasetFormats:
    def test_jsonl_dataset_works_with_ablation_sweep(self, tmp_path: Path) -> None:
        dataset = tmp_path / "ds.jsonl"
        dataset.write_text(
            '{"question": "Who invented the telephone?", '
            '"answers": ["Alexander Graham Bell"], '
            '"evidence_passages": []}\n',
            encoding="utf-8",
        )
        output_json = tmp_path / "out.json"

        class FakeWatson:
            def __init__(self, config: FeatureConfig) -> None: ...

            def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
                del question, verbose
                return _final_answer("Alexander Graham Bell", 0.9)

        with patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson):
            results, _ = run_benchmark_profiles(
                dataset_path=str(dataset),
                config=FeatureConfig.baseline(),
                output_json_path=str(output_json),
                ablation_sweep=True,
            )

        assert len(results) == len(build_ablation_profiles())

    def test_samples_dict_format_with_samples_key(self, tmp_path: Path) -> None:
        dataset = tmp_path / "ds.json"
        dataset.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "question": "What is 2+2?",
                            "answers": ["4"],
                            "evidence_passages": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_json = tmp_path / "out.json"

        class FakeWatson:
            def __init__(self, config: FeatureConfig) -> None: ...

            def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
                del question, verbose
                return _final_answer("4", 0.9)

        with patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson):
            results, _ = run_benchmark_profiles(
                dataset_path=str(dataset),
                config=FeatureConfig.baseline(),
                output_json_path=str(output_json),
                ablation_sweep=True,
            )

        assert len(results) == len(build_ablation_profiles())
