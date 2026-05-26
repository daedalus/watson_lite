import json
from pathlib import Path
from unittest.mock import patch

from watson_lite.core.cache import Cache
from watson_lite.core.config import OPTIONAL_FEATURES, FeatureConfig
from watson_lite.core.models import AnswerDiagnostics, FinalAnswer
from watson_lite.evaluation.benchmark_runner import (
    RegressionThresholds,
    build_ablation_profiles,
    run_benchmark_profiles,
)


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


def test_build_ablation_profiles_contains_expected_profiles() -> None:
    profiles = build_ablation_profiles()
    assert "baseline" in profiles
    assert "minimal" in profiles
    assert len(profiles) == 2 + (2 * len(OPTIONAL_FEATURES))


def test_run_benchmark_profiles_outputs_files(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    output_json = tmp_path / "out.json"
    output_csv = tmp_path / "out.csv"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "What is the capital of France?",
                    "answers": ["Paris"],
                    "evidence_passages": ["capital city of France"],
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeWatson:
        def __init__(self, config: FeatureConfig) -> None:
            self.config = config

        def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
            del question, verbose
            return _final_answer("Paris", 0.9)

    with (
        patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson),
        patch("watson_lite.evaluation.benchmark_runner._get_answer_cache") as mock_cache,
    ):
        mock_cache.return_value = Cache(db_path=str(tmp_path / "cache_br1.sqlite3"))
        results, regressions = run_benchmark_profiles(
            dataset_path=str(dataset),
            config=FeatureConfig.baseline(),
            output_json_path=str(output_json),
            output_csv_path=str(output_csv),
            ablation_sweep=True,
            regression_check=True,
        )

    assert len(results) == 2 + (2 * len(OPTIONAL_FEATURES))
    assert regressions == []
    assert output_json.exists()
    assert output_csv.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "confidence_calibration_kl_divergence" in payload["results"][0]["metrics"]
    assert "confidence_calibration_js_divergence" in payload["results"][0]["metrics"]
    header = output_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "confidence_calibration_kl_divergence" in header
    assert "confidence_calibration_js_divergence" in header


def test_run_benchmark_profiles_detects_regression(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    output_json = tmp_path / "out.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question": "What is the capital of France?",
                    "answers": ["Paris"],
                    "evidence_passages": ["capital city of France"],
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeWatson:
        def __init__(self, config: FeatureConfig) -> None:
            self.config = config

        def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
            del question, verbose
            if self.config == FeatureConfig.baseline():
                return _final_answer("Paris", 0.9)
            return _final_answer("London", 0.8, passages=["Berlin is in Germany."])

    with (
        patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson),
        patch("watson_lite.evaluation.benchmark_runner._get_answer_cache") as mock_cache,
    ):
        mock_cache.return_value = Cache(db_path=str(tmp_path / "cache_br2.sqlite3"))
        _, regressions = run_benchmark_profiles(
            dataset_path=str(dataset),
            config=FeatureConfig.minimal(),
            output_json_path=str(output_json),
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
