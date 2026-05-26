"""Tests for watson_lite.evaluation.benchmarks.triviaqa."""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from watson_lite.evaluation.benchmarks.triviaqa import (
    _convert_triviaqa,
    download_triviaqa,
    main,
)

_RAW_TRIVIAQA = {
    "Data": [
        {
            "Question": "Who wrote Hamlet?",
            "Answer": {
                "Aliases": ["Shakespeare", "William Shakespeare", ""],
                "NormalizedAliases": ["shakespeare", "william shakespeare"],
            },
        },
        {
            "Question": "What is 2+2?",
            "Answer": {
                "Aliases": ["Four", "4"],
            },
        },
        {
            # Missing question → should be skipped
            "Question": "",
            "Answer": {"Aliases": ["something"]},
        },
        {
            # No answers → should be skipped
            "Question": "No answer?",
            "Answer": {"Aliases": []},
        },
    ]
}


class TestConvertTriviaQA:
    def _raw_bytes(self, data: dict | None = None) -> bytes:
        return json.dumps(data or _RAW_TRIVIAQA).encode("utf-8")

    def test_basic_conversion(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        samples = _convert_triviaqa(self._raw_bytes(), str(output))

        assert len(samples) == 2
        assert samples[0]["question"] == "Who wrote Hamlet?"
        assert "Shakespeare" in samples[0]["answers"]
        # Empty string alias should be stripped
        assert "" not in samples[0]["answers"]

    def test_deduplicates_aliases_preserving_order(self, tmp_path: Path) -> None:
        data = {
            "Data": [
                {
                    "Question": "Q?",
                    "Answer": {"Aliases": ["A", "B", "A"]},
                }
            ]
        }
        output = tmp_path / "out.json"
        samples = _convert_triviaqa(json.dumps(data).encode(), str(output))

        assert samples[0]["answers"].count("A") == 1

    def test_max_samples_limits_output(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        samples = _convert_triviaqa(self._raw_bytes(), str(output), max_samples=1)

        assert len(samples) == 1

    def test_writes_output_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        _convert_triviaqa(self._raw_bytes(), str(output))

        data = json.loads(output.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 2

    def test_evidence_passages_is_empty(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        samples = _convert_triviaqa(self._raw_bytes(), str(output))

        assert all(s["evidence_passages"] == [] for s in samples)

    def test_null_answer_data_skipped(self, tmp_path: Path) -> None:
        data = {
            "Data": [
                {"Question": "Q?", "Answer": None},
                {"Question": "Q2?", "Answer": {"Aliases": ["A"]}},
            ]
        }
        output = tmp_path / "out.json"
        samples = _convert_triviaqa(json.dumps(data).encode(), str(output))
        assert len(samples) == 1


class TestDownloadTriviaQA:
    def _make_archive(self, tmp_path: Path, member_path: str, content: bytes) -> Path:
        archive = tmp_path / "triviaqa-rc.tar.gz"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name=member_path)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        archive.write_bytes(buf.getvalue())
        return archive

    def test_raises_on_unknown_split(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            download_triviaqa(str(tmp_path / "out.json"), split="invalid")

    def test_dev_split(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        raw_bytes = json.dumps(_RAW_TRIVIAQA).encode("utf-8")

        with (
            patch(
                "watson_lite.evaluation.benchmarks.triviaqa.download_with_resume"
            ) as mock_dl,
            patch(
                "watson_lite.evaluation.benchmarks.triviaqa.stream_extract_tar_member",
                return_value=raw_bytes,
            ),
        ):
            mock_dl.side_effect = lambda url, path, **kw: path

            samples = download_triviaqa(str(output), split="dev", download_dir=str(tmp_path))

        assert isinstance(samples, list)
        assert len(samples) == 2

    def test_test_split(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        raw_bytes = json.dumps(_RAW_TRIVIAQA).encode("utf-8")

        with (
            patch(
                "watson_lite.evaluation.benchmarks.triviaqa.download_with_resume"
            ) as mock_dl,
            patch(
                "watson_lite.evaluation.benchmarks.triviaqa.stream_extract_tar_member",
                return_value=raw_bytes,
            ),
        ):
            mock_dl.side_effect = lambda url, path, **kw: path

            samples = download_triviaqa(str(output), split="test", download_dir=str(tmp_path))

        assert isinstance(samples, list)


class TestTriviaQAMain:
    def test_main_runs_without_run_flag(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        with (
            patch("sys.argv", ["prog", "--output", str(output)]),
            patch(
                "watson_lite.evaluation.benchmarks.triviaqa.download_triviaqa",
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
                "watson_lite.evaluation.benchmarks.triviaqa.download_triviaqa",
                return_value=[],
            ),
            patch(
                "watson_lite.evaluation.benchmark_runner.run_benchmark_profiles",
                return_value=([], []),
            ) as mock_run,
        ):
            main()

        mock_run.assert_called_once()

    def test_main_test_split(self, tmp_path: Path) -> None:
        output = tmp_path / "out.json"
        with (
            patch("sys.argv", ["prog", "--output", str(output), "--split", "test"]),
            patch(
                "watson_lite.evaluation.benchmarks.triviaqa.download_triviaqa",
                return_value=[],
            ) as mock_dl,
        ):
            main()

        call_kwargs = mock_dl.call_args
        assert "test" in str(call_kwargs)
