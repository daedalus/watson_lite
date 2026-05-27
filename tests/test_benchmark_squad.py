"""Tests for watson_lite.evaluation.benchmarks.squad."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from watson_lite.evaluation.benchmarks.squad import (
    _convert_squad,
    download_squad,
    main,
)

_RAW_SQUAD = {
    "data": [
        {
            "title": "Test Article",
            "paragraphs": [
                {
                    "context": "The sky is blue.",
                    "qas": [
                        {
                            "id": "1",
                            "question": "What color is the sky?",
                            "is_impossible": False,
                            "answers": [{"text": "blue", "answer_start": 11}],
                        },
                        {
                            "id": "2",
                            "question": "Impossible Q",
                            "is_impossible": True,
                            "answers": [],
                        },
                        {
                            "id": "3",
                            "question": "Empty answers?",
                            "is_impossible": False,
                            "answers": [],
                        },
                    ],
                },
                {
                    "context": "Water is wet.",
                    "qas": [
                        {
                            "id": "4",
                            "question": "What is water?",
                            "is_impossible": False,
                            "answers": [
                                {"text": "wet", "answer_start": 9},
                                {"text": "wet", "answer_start": 9},  # duplicate
                            ],
                        },
                    ],
                },
            ],
        }
    ]
}


class TestConvertSquad:
    def test_basic_conversion(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "squad.json"
        raw_path.write_text(json.dumps(_RAW_SQUAD), encoding="utf-8")
        output_path = tmp_path / "output.json"

        samples = _convert_squad(raw_path, str(output_path))

        assert len(samples) == 2
        assert samples[0]["question"] == "What color is the sky?"
        assert "blue" in samples[0]["answers"]
        assert samples[0]["evidence_passages"] == ["The sky is blue."]

    def test_deduplicates_answers(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "squad.json"
        raw_path.write_text(json.dumps(_RAW_SQUAD), encoding="utf-8")
        output_path = tmp_path / "output.json"

        samples = _convert_squad(raw_path, str(output_path))

        water_sample = next(s for s in samples if s["question"] == "What is water?")
        assert water_sample["answers"].count("wet") == 1

    def test_max_samples_limits_output(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "squad.json"
        raw_path.write_text(json.dumps(_RAW_SQUAD), encoding="utf-8")
        output_path = tmp_path / "output.json"

        samples = _convert_squad(raw_path, str(output_path), max_samples=1)

        assert len(samples) == 1

    def test_writes_output_file(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "squad.json"
        raw_path.write_text(json.dumps(_RAW_SQUAD), encoding="utf-8")
        output_path = tmp_path / "output.json"

        _convert_squad(raw_path, str(output_path))

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 2

    def test_empty_dataset(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "squad.json"
        raw_path.write_text(json.dumps({"data": []}), encoding="utf-8")
        output_path = tmp_path / "output.json"

        samples = _convert_squad(raw_path, str(output_path))

        assert samples == []

    def test_max_samples_stops_at_article_boundary(self, tmp_path: Path) -> None:
        # Large dataset to exercise the max_samples break at article level
        many_articles = {
            "data": [
                {
                    "title": f"Article {i}",
                    "paragraphs": [
                        {
                            "context": "Some context.",
                            "qas": [
                                {
                                    "id": str(i),
                                    "question": f"Question {i}?",
                                    "is_impossible": False,
                                    "answers": [{"text": "answer", "answer_start": 0}],
                                }
                            ],
                        }
                    ],
                }
                for i in range(5)
            ]
        }
        raw_path = tmp_path / "squad.json"
        raw_path.write_text(json.dumps(many_articles), encoding="utf-8")
        output_path = tmp_path / "output.json"

        samples = _convert_squad(raw_path, str(output_path), max_samples=3)

        assert len(samples) == 3


class TestDownloadSquad:
    def test_uses_provided_download_dir(self, tmp_path: Path) -> None:
        dl_dir = tmp_path / "downloads"
        output_path = tmp_path / "squad_out.json"

        with patch(
            "watson_lite.evaluation.benchmarks.squad.download_with_resume"
        ) as mock_dl:
            raw_file = dl_dir / "squad_v2_dev.json"
            mock_dl.side_effect = lambda url, path, **kw: (
                path.parent.mkdir(parents=True, exist_ok=True)
                or path.write_text(json.dumps(_RAW_SQUAD), encoding="utf-8")
                or path
            )
            samples = download_squad(str(output_path), download_dir=str(dl_dir))

        assert isinstance(samples, list)
        assert len(samples) > 0
        mock_dl.assert_called_once()

    def test_uses_temp_dir_when_no_download_dir(self, tmp_path: Path) -> None:
        output_path = tmp_path / "squad_out.json"

        with (
            patch(
                "watson_lite.evaluation.benchmarks.squad.download_with_resume"
            ) as mock_dl,
            patch(
                "watson_lite.evaluation.benchmarks.squad.tempfile.mkdtemp",
                return_value=str(tmp_path / "tmpdir"),
            ),
        ):
            (tmp_path / "tmpdir").mkdir()
            raw_path = tmp_path / "tmpdir" / "squad_v2_dev.json"
            raw_path.write_text(json.dumps(_RAW_SQUAD), encoding="utf-8")
            mock_dl.side_effect = lambda url, path, **kw: path

            samples = download_squad(str(output_path))

        assert isinstance(samples, list)


class TestSquadMain:
    def test_main_runs_without_run_flag(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        with (
            patch("sys.argv", ["prog", "--output", str(output)]),
            patch(
                "watson_lite.evaluation.benchmarks.squad.download_squad",
                return_value=[
                    {"question": "Q?", "answers": ["A"], "evidence_passages": []}
                ],
            ) as mock_ds,
        ):
            main()

        mock_ds.assert_called_once()

    def test_main_with_run_flag(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        output.write_text(json.dumps([]), encoding="utf-8")
        with (
            patch("sys.argv", ["prog", "--output", str(output), "--run"]),
            patch(
                "watson_lite.evaluation.benchmarks.squad.download_squad",
                return_value=[],
            ),
            patch(
                "watson_lite.evaluation.benchmark_runner.run_benchmark_profiles",
                return_value=([], []),
            ) as mock_run,
        ):
            main()

        mock_run.assert_called_once()

    def test_main_with_max_samples(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        with (
            patch("sys.argv", ["prog", "--output", str(output), "--max-samples", "10"]),
            patch(
                "watson_lite.evaluation.benchmarks.squad.download_squad",
                return_value=[],
            ) as mock_ds,
        ):
            main()

        call_kwargs = mock_ds.call_args
        assert call_kwargs.kwargs.get("max_samples") == 10 or call_kwargs.args[1] == 10
