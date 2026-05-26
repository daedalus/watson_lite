import json
from pathlib import Path
from unittest.mock import patch

from watson_lite.core.cache import Cache
from watson_lite.core.config import FeatureConfig
from watson_lite.core.models import AnswerDiagnostics, FinalAnswer
from watson_lite.evaluation.benchmark_runner import (
    load_benchmark_dataset,
    run_benchmark_profiles,
)

BENCHMARK_DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "smoke.json"


def _answer_for_question(question: str) -> FinalAnswer:
    answers = {
        "Who designed the Eiffel Tower?": "Gustave Eiffel",
        "What is the capital of France?": "Paris",
        "When did the Apollo 11 mission land on the Moon?": "1969",
    }
    passages = {
        "Who designed the Eiffel Tower?": ["The tower was designed by Gustave Eiffel."],
        "What is the capital of France?": ["Paris is the capital city of France."],
        "When did the Apollo 11 mission land on the Moon?": [
            "Apollo 11 landed on the Moon in 1969."
        ],
    }
    return FinalAnswer(
        answer=answers[question],
        confidence=0.95,
        source="benchmark",
        url="https://example.org",
        supporting_passages=passages[question],
        confidence_breakdown={"graph_corroboration": 0.0, "type_coercion": 0.0},
        diagnostics=AnswerDiagnostics(
            total_latency_s=0.1,
            top_retrieved_passages=passages[question],
        ),
    )


def test_checked_in_benchmark_dataset_is_valid() -> None:
    samples = load_benchmark_dataset(str(BENCHMARK_DATASET))

    assert len(samples) == 3
    assert all(sample.question for sample in samples)
    assert all(sample.answers for sample in samples)


def test_checked_in_benchmark_dataset_runs_regression_smoke(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "benchmark.json"

    class FakeWatson:
        def __init__(self, config: FeatureConfig) -> None:
            self.config = config

        def answer(self, question: str, verbose: bool = False) -> FinalAnswer:
            _ = verbose
            return _answer_for_question(question)

    with (
        patch("watson_lite.evaluation.benchmark_runner.WatsonLite", FakeWatson),
        patch(
            "watson_lite.evaluation.benchmark_runner._get_answer_cache"
        ) as mock_cache,
    ):
        mock_cache.return_value = Cache(db_path=str(tmp_path / "cache_bd.sqlite3"))
        results, regressions = run_benchmark_profiles(
            dataset_path=str(BENCHMARK_DATASET),
            config=FeatureConfig.baseline(),
            output_json_path=str(output_json),
            regression_check=True,
        )

    assert results
    assert regressions == []
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "results" in payload
