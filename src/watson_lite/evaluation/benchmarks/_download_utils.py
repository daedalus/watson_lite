from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from pathlib import Path


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 80


def progress_bar(description: str, current: int, total: int) -> None:
    if total <= 0:
        return
    term_w = _term_width()
    bar_w = max(10, term_w - len(description) - 12)
    fraction = min(current / total, 1.0)
    filled = int(bar_w * fraction)
    bar = "█" * filled + "░" * (bar_w - filled)
    pct = fraction * 100
    sys.stdout.write(f"\r{description}: [{bar}] {pct:5.1f}%")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _remote_size(url: str, timeout: int = 30) -> int | None:
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        raw = resp.headers.get("Content-Length")
        return int(raw) if raw else None
    except (requests.RequestException, ValueError, TypeError):
        return None


def download_with_resume(
    url: str,
    dest: Path,
    *,
    label: str | None = None,
    timeout: int = 600,
) -> Path:
    label = label or dest.name

    if dest.exists() and dest.stat().st_size > 0:
        remote_len = _remote_size(url)
        if remote_len is not None and dest.stat().st_size >= remote_len:
            print(
                f"{label}: already downloaded ({dest.stat().st_size} bytes), skipping"
            )
            return dest

    part_path = dest.with_suffix(dest.suffix + ".part")
    headers: dict[str, str] = {}
    mode = "wb"
    start_byte = 0

    if part_path.exists():
        start_byte = part_path.stat().st_size
        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"
            mode = "ab"

    resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
    if resp.status_code == 416:
        start_byte = 0
        headers.pop("Range", None)
        mode = "wb"
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)

    raw_total = 0
    try:
        raw_total = int(
            resp.raw.headers.get("Content-Length", 0)  # type: ignore[union-attr]
        )
    except (TypeError, ValueError, AttributeError):
        raw_total = 0
    total = int(resp.headers.get("Content-Length", raw_total))
    if start_byte > 0 and resp.status_code == 206:
        total += start_byte

    resp.raise_for_status()

    with open(part_path, mode) as f:
        downloaded = start_byte
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                progress_bar(label, downloaded, total)

    part_path.rename(dest)
    return dest


def stream_extract_tar_member(
    archive_path: Path,
    member_path: str,
) -> bytes:
    import tarfile

    with tarfile.open(archive_path, "r:gz") as tar:
        member = tar.getmember(member_path)
        f = tar.extractfile(member)
        if f is None:
            msg = f"Could not extract {member_path} from {archive_path}"
            raise RuntimeError(msg)
        return f.read()
