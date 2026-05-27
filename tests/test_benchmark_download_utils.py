"""Tests for watson_lite.evaluation.benchmarks._download_utils."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from watson_lite.evaluation.benchmarks._download_utils import (
    _remote_size,
    _term_width,
    download_with_resume,
    progress_bar,
    stream_extract_tar_member,
)


class TestTermWidth:
    def test_returns_columns(self) -> None:
        with patch("os.get_terminal_size") as mock_ts:
            mock_ts.return_value = MagicMock(columns=120)
            assert _term_width() == 120

    def test_fallback_on_os_error(self) -> None:
        with patch("os.get_terminal_size", side_effect=OSError):
            assert _term_width() == 80

    def test_fallback_on_value_error(self) -> None:
        with patch("os.get_terminal_size", side_effect=ValueError):
            assert _term_width() == 80


class TestProgressBar:
    def test_zero_total_returns_immediately(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        progress_bar("test", 0, 0)
        assert capsys.readouterr().out == ""

    def test_intermediate_progress(self, capsys: pytest.CaptureFixture[str]) -> None:
        progress_bar("Downloading", 50, 100)
        out = capsys.readouterr().out
        assert "Downloading" in out
        assert "50.0%" in out

    def test_complete_progress_adds_newline(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        progress_bar("Downloading", 100, 100)
        out = capsys.readouterr().out
        assert out.endswith("\n")
        assert "100.0%" in out

    def test_over_total_clamps(self, capsys: pytest.CaptureFixture[str]) -> None:
        progress_bar("X", 200, 100)
        out = capsys.readouterr().out
        assert "100.0%" in out

    def test_negative_total_returns_immediately(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        progress_bar("test", 5, -1)
        assert capsys.readouterr().out == ""


class TestRemoteSize:
    def test_returns_content_length(self) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "12345"}
        with patch(
            "watson_lite.evaluation.benchmarks._download_utils.requests.head",
            return_value=mock_resp,
        ):
            assert _remote_size("http://example.com/file") == 12345

    def test_returns_none_when_no_header(self) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = {}
        with patch(
            "watson_lite.evaluation.benchmarks._download_utils.requests.head",
            return_value=mock_resp,
        ):
            assert _remote_size("http://example.com/file") is None

    def test_returns_none_on_request_exception(self) -> None:
        import requests

        with patch(
            "watson_lite.evaluation.benchmarks._download_utils.requests.head",
            side_effect=requests.RequestException,
        ):
            assert _remote_size("http://example.com/file") is None

    def test_returns_none_on_value_error(self) -> None:
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "not-a-number"}
        with patch(
            "watson_lite.evaluation.benchmarks._download_utils.requests.head",
            return_value=mock_resp,
        ):
            assert _remote_size("http://example.com/file") is None


class TestDownloadWithResume:
    def _make_response(
        self,
        content: bytes = b"hello world",
        status_code: int = 200,
        content_length: str | None = None,
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = {}
        if content_length is not None:
            resp.headers["Content-Length"] = content_length
        resp.raw = MagicMock()
        resp.raw.headers = {}
        chunk_size = 4096
        chunks = [
            content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
        ] or [b""]
        resp.iter_content.return_value = iter(chunks)
        resp.raise_for_status = MagicMock()
        return resp

    def test_skips_when_already_downloaded(self, tmp_path: Path) -> None:
        dest = tmp_path / "file.bin"
        content = b"already downloaded"
        dest.write_bytes(content)

        with patch(
            "watson_lite.evaluation.benchmarks._download_utils._remote_size",
            return_value=len(content),
        ):
            result = download_with_resume("http://example.com/file", dest)

        assert result == dest
        assert dest.read_bytes() == content

    def test_downloads_new_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "file.bin"
        content = b"new content"
        resp = self._make_response(content, content_length=str(len(content)))

        with patch(
            "watson_lite.evaluation.benchmarks._download_utils.requests.get",
            return_value=resp,
        ):
            result = download_with_resume("http://example.com/file", dest)

        assert result == dest
        assert dest.read_bytes() == content

    def test_resumes_partial_download(self, tmp_path: Path) -> None:
        dest = tmp_path / "file.bin"
        partial = tmp_path / "file.bin.part"
        partial_content = b"partial"
        partial.write_bytes(partial_content)

        remaining = b" rest"
        resp = MagicMock()
        resp.status_code = 206
        resp.headers = {"Content-Length": str(len(remaining))}
        resp.raw = MagicMock()
        resp.raw.headers = {}
        resp.iter_content.return_value = iter([remaining])
        resp.raise_for_status = MagicMock()

        with patch(
            "watson_lite.evaluation.benchmarks._download_utils.requests.get",
            return_value=resp,
        ):
            result = download_with_resume("http://example.com/file", dest)

        assert result == dest
        assert dest.read_bytes() == partial_content + remaining

    def test_handles_416_range_not_satisfiable(self, tmp_path: Path) -> None:
        dest = tmp_path / "file.bin"
        partial = tmp_path / "file.bin.part"
        partial.write_bytes(b"stale partial")

        content = b"fresh content"
        resp416 = MagicMock()
        resp416.status_code = 416

        resp_ok = self._make_response(content, content_length=str(len(content)))

        call_count = [0]

        def side_effect(url: str, **kwargs: object) -> MagicMock:
            call_count[0] += 1
            return resp416 if call_count[0] == 1 else resp_ok

        with patch(
            "watson_lite.evaluation.benchmarks._download_utils.requests.get",
            side_effect=side_effect,
        ):
            result = download_with_resume("http://example.com/file", dest)

        assert result == dest
        assert dest.read_bytes() == content

    def test_uses_label_fallback(self, tmp_path: Path) -> None:
        dest = tmp_path / "data.bin"
        content = b"xyz"
        resp = self._make_response(content)

        with patch(
            "watson_lite.evaluation.benchmarks._download_utils.requests.get",
            return_value=resp,
        ):
            result = download_with_resume("http://example.com/file", dest)

        assert result == dest


class TestStreamExtractTarMember:
    def test_extracts_member(self, tmp_path: Path) -> None:
        archive = tmp_path / "test.tar.gz"
        member_path = "qa/data.json"
        member_content = b'{"key": "value"}'

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name=member_path)
            info.size = len(member_content)
            tar.addfile(info, io.BytesIO(member_content))
        archive.write_bytes(buf.getvalue())

        result = stream_extract_tar_member(archive, member_path)
        assert result == member_content

    def test_raises_on_missing_member(self, tmp_path: Path) -> None:
        archive = tmp_path / "test.tar.gz"

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="other/file.txt")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"data"))
        archive.write_bytes(buf.getvalue())

        with pytest.raises(KeyError):
            stream_extract_tar_member(archive, "qa/missing.json")

    def test_raises_on_non_extractable_member(self, tmp_path: Path) -> None:
        archive = tmp_path / "test.tar.gz"
        # Create a directory entry (not a regular file), which extractfile returns None for
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="qa/dir")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        archive.write_bytes(buf.getvalue())

        with pytest.raises(RuntimeError, match="Could not extract"):
            stream_extract_tar_member(archive, "qa/dir")
