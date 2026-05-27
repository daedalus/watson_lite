from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from watson_lite.core.models import Passage

logger = logging.getLogger(__name__)
_NON_WORD_RE = re.compile(r"\W+")
_TEXT_KEYS = ("text", "content", "body", "passage", "snippet", "document")
_SOURCE_KEYS = ("source", "title", "name")
_URL_KEYS = ("url", "source_url")
_OFFLINE_DATASET_DIR_ENV = "WATSON_LITE_OFFLINE_DATASET_DIR"
_OFFLINE_PATH_SUFFIX = "_PATH"
_OFFLINE_ENV_PREFIX = "WATSON_LITE_OFFLINE_"
_OFFLINE_CORPUS_CACHE: dict[str, list[Passage]] = {}


def _tokenize(value: str) -> set[str]:
    return {
        token
        for token in _NON_WORD_RE.sub(" ", value.lower()).split()
        if token and len(token) > 1
    }


def _dataset_env_name(dataset_name: str) -> str:
    normalized = _NON_WORD_RE.sub("_", dataset_name.upper()).strip("_")
    return f"{_OFFLINE_ENV_PREFIX}{normalized}{_OFFLINE_PATH_SUFFIX}"


def _resolve_dataset_path(dataset_name: str, base_dir: str | None) -> Path | None:
    explicit_path = os.getenv(_dataset_env_name(dataset_name), "").strip()
    if explicit_path:
        return Path(explicit_path)

    resolved_base = (base_dir or "").strip() or os.getenv(
        _OFFLINE_DATASET_DIR_ENV, ""
    ).strip()
    if not resolved_base:
        return None
    return Path(resolved_base) / f"{dataset_name}.jsonl"


def _extract_text(payload: dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_field(payload: dict[str, Any], keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _load_json_payload(path: Path) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        logger.warning("Offline dataset parse failed for '%s': %s", path, err)
        return []
    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]
    if isinstance(loaded, dict):
        rows = loaded.get("rows") or loaded.get("documents") or loaded.get("data")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _load_jsonl_payload(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                rows.append(parsed)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        logger.warning("Offline dataset parse failed for '%s': %s", path, err)
        return []
    return rows


def _load_offline_corpus(path: Path, dataset_name: str) -> list[Passage]:
    cache_key = str(path.resolve())
    cached = _OFFLINE_CORPUS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not path.exists():
        logger.warning(
            "Offline dataset path does not exist for '%s': %s", dataset_name, path
        )
        _OFFLINE_CORPUS_CACHE[cache_key] = []
        return []

    if path.suffix.lower() == ".json":
        rows = _load_json_payload(path)
    else:
        rows = _load_jsonl_payload(path)

    passages: list[Passage] = []
    for row in rows:
        text = _extract_text(row)
        if not text:
            continue
        source = _extract_field(row, _SOURCE_KEYS, default=dataset_name)
        url = _extract_field(row, _URL_KEYS, default=f"file://{path}")
        passages.append(Passage(text=text, source=source, url=url))

    _OFFLINE_CORPUS_CACHE[cache_key] = passages
    return passages


def fetch_offline_dataset_passages(
    query: str,
    *,
    top_k: int,
    dataset_name: str,
    base_dir: str | None = None,
) -> list[Passage]:
    """Fetch passages from a local JSON/JSONL corpus for an offline dataset plugin."""
    dataset_path = _resolve_dataset_path(dataset_name, base_dir)
    if dataset_path is None:
        logger.warning(
            "Offline dataset '%s' is not configured. Set %s or %s.",
            dataset_name,
            _OFFLINE_DATASET_DIR_ENV,
            _dataset_env_name(dataset_name),
        )
        return []

    passages = _load_offline_corpus(dataset_path, dataset_name)
    if not passages:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return passages[:top_k]

    scored: list[tuple[float, Passage]] = []
    for passage in passages:
        passage_terms = _tokenize(passage.text)
        overlap = query_terms.intersection(passage_terms)
        if not overlap:
            continue
        score = len(overlap) / len(query_terms)
        scored.append((score, passage))

    scored.sort(key=lambda item: item[0], reverse=True)
    ranked: list[Passage] = []
    for rank, (score, passage) in enumerate(scored[:top_k], start=1):
        ranked.append(
            Passage(
                text=passage.text,
                source=passage.source,
                url=passage.url,
                score=score,
                rank=rank,
            )
        )
    return ranked
