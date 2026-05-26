"""Tests for watson_lite.evaluation.benchmarks.natural_questions."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from watson_lite.evaluation.benchmarks.natural_questions import (
    _convert_nq,
    download_natural_questions,
    main,
)

_NQ_JSONL = textwrap.dedent("""\
    {"question": "Who is Alan Turing?", "answer": ["mathematician", "computer scientist"]}
    {"question": "Where is Paris?", "answer": ["France"]}
    {"question": "  ", "answer": ["something"]}
    {"question": "No answer?", "answer": []}
    {"question": "String answer?", "answer": "single"}
    \n
""")


class TestConvertNQ:
    def test_basic_conversion(self, tmp_path: Path) -> None:
        raw = tmp_path / "nq.jsonl"
        raw.write_text(_NQ_JSONL, encoding="utf-8")
        output = tmp_path / "out.json"

        samples = _convert_nq(raw, str(output))

        # Only valid non-empty records should be included
        assert len(samples) == 3
        assert samples[0]["question"] == "Who is Alan Turing?"
        assert "mathematician" in samples[0]["answers"]
        assert "computer scientist" in samples[0]["answers"]

    def test_string_answer_wrapped_in_list(self, tmp_path: Path) -> None:
        raw = tmp_path / "nq.jsonl"
        raw.write_text(_NQ_JSONL, encoding="utf-8")
        output = tmp_path / "out.json"

        samples = _convert_nq(raw, str(output))

        string_sample = next(s for s in samples if s["question"] == "String answer?")
        assert string_sample["answers"] == ["single"]

    def test_max_samples_limits_output(self, tmp_path: Path) -> None:
        raw = tmp_path / "nq.jsonl"
        raw.write_text(_NQ_JSONL, encoding="utf-8")
        output = tmp_path / "out.json"

        samples = _convert_nq(raw, str(output), max_samples=1)

        assert len(samples) == 1

    def test_writes_output_file(self, tmp_path: Path) -> None:
        raw = tmp_path / "nq.jsonl"
        raw.write_text(_NQ_JSONL, encoding="utf-8")
        output = tmp_path / "out.json"

        _convert_nq(raw, str(output))

        data = json.loads(output.read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_evidence_passages_is_empty(self, tmp_path: Path) -> None:
        raw = tmp_path / "nq.jsonl"
        raw.write_text(_NQ_JSONL, encoding="utf-8")
        output = tmp_path / "out.json"

        samples = _convert_nq(raw, str(output))

        assert all(s["evidence_passages"] == [] for s in samples)

    def test_deduplicates_answers(self, tmp_path: Path) -> None:
        jsonl = '{"question": "Q?", "answer": ["A", "B", "A"]}\n'
        raw = tmp_path / "nq.jsonl"
        raw.write_text(jsonl, encoding="utf-8")
        output = tmp_path / "out.json"

        samples = _convert_nq(raw, str(output))

        assert samples[0]["answers"].count("A") == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        raw = tmp_path / "nq.jsonl"
        raw.write_text("", encoding="utf-8")
        output = tmp_path / "out.json"

        samples = _convert_nq(raw, str(output))

        assert samples == []


class TestDownloadNaturalQuestions:
    def test_raises_on_unknown_split(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            download_natural_questions(str(tmp_path / "out.json"), split="invalid")

    def test_dev_split_with_download_dir(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        dl_dir = tmp_path / "downloads"

        with patch(
            "watson_lite.evaluation.benchmarks.natural_questions.download_with_resume"
        ) as mock_dl:
            raw_path = dl_dir / "nq_open_dev.jsonl"

            def _side_effect(url: str, path: Path, **kw: object) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    '{"question": "Who?", "answer": ["Someone"]}\n', encoding="utf-8"
                )
                return path

            mock_dl.side_effect = _side_effect

            samples = download_natural_questions(
                str(output), split="dev", download_dir=str(dl_dir)
            )

        assert isinstance(samples, list)
        assert len(samples) == 1
        mock_dl.assert_called_once()

    def test_train_split_without_download_dir(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"

        with patch(
            "watson_lite.evaluation.benchmarks.natural_questions.download_with_resume"
        ) as mock_dl, patch(
            "watson_lite.evaluation.benchmarks.natural_questions.tempfile.mkdtemp",
            return_value=str(tmp_path / "tmpdir"),
        ):
            (tmp_path / "tmpdir").mkdir()
            raw_path = tmp_path / "tmpdir" / "nq_open_train.jsonl"
            raw_path.write_text(
                '{"question": "When?", "answer": ["1999"]}\n', encoding="utf-8"
            )
            mock_dl.side_effect = lambda url, path, **kw: path

            samples = download_natural_questions(str(output), split="train")

        assert isinstance(samples, list)
        assert len(samples) == 1


class TestNaturalQuestionsMain:
    def test_main_runs_without_run_flag(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        with (
            patch("sys.argv", ["prog", "--output", str(output)]),
            patch(
                "watson_lite.evaluation.benchmarks.natural_questions.download_natural_questions",
                return_value=[],
            ) as mock_dl,
        ):
            main()

        mock_dl.assert_called_once()

    def test_main_with_run_flag(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        output.write_text(json.dumps([]), encoding="utf-8")
        with (
            patch("sys.argv", ["prog", "--output", str(output), "--run"]),
            patch(
                "watson_lite.evaluation.benchmarks.natural_questions.download_natural_questions",
                return_value=[],
            ),
            patch(
                "watson_lite.evaluation.benchmark_runner.run_benchmark_profiles",
                return_value=([], []),
            ) as mock_run,
        ):
            main()

        mock_run.assert_called_once()

    def test_main_train_split(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        with (
            patch("sys.argv", ["prog", "--output", str(output), "--split", "train"]),
            patch(
                "watson_lite.evaluation.benchmarks.natural_questions.download_natural_questions",
                return_value=[],
            ) as mock_dl,
        ):
            main()

        call_kwargs = mock_dl.call_args
        assert "train" in str(call_kwargs)

    def test_main_with_download_dir(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        dl_dir = tmp_path / "downloads"
        with (
            patch(
                "sys.argv",
                ["prog", "--output", str(output), "--download-dir", str(dl_dir)],
            ),
            patch(
                "watson_lite.evaluation.benchmarks.natural_questions.download_natural_questions",
                return_value=[],
            ) as mock_dl,
        ):
            main()

        mock_dl.assert_called_once()
