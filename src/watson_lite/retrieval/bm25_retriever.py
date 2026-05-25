import copy
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import bm25s
import requests

from watson_lite.core.cache import get_cache, is_cache_miss
from watson_lite.core.models import Passage

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIBOOKS_API = "https://en.wikibooks.org/w/api.php"
WIKI_SEARCH_LIMIT = 5
ELASTICSEARCH_URL_ENV = "WATSON_LITE_ELASTICSEARCH_URL"
ELASTICSEARCH_INDEX_ENV = "WATSON_LITE_ELASTICSEARCH_INDEX"
ELASTICSEARCH_API_KEY_ENV = "WATSON_LITE_ELASTICSEARCH_API_KEY"
HUGGINGFACE_DATASET_ENV = "WATSON_LITE_HUGGINGFACE_DATASET"
HUGGINGFACE_CONFIG_ENV = "WATSON_LITE_HUGGINGFACE_CONFIG"
HUGGINGFACE_SPLIT_ENV = "WATSON_LITE_HUGGINGFACE_SPLIT"
HUGGINGFACE_TOKEN_ENV = "WATSON_LITE_HUGGINGFACE_TOKEN"
ELASTICSEARCH_TIMEOUT_SECONDS = 10
HUGGINGFACE_DATASET_SERVER_SEARCH = "https://datasets-server.huggingface.co/search"
# Keep chunks comfortably below the reader's context budget while leaving room
# for sentence overlap, which reduces edge-truncation during extraction.
CHUNK_SIZE = 180
CHUNK_OVERLAP_SENTENCES = 1
MIN_CHUNK_WORDS = 20
FALLBACK_CHUNK_STEP = CHUNK_SIZE // 2
_NEGATIVE_CACHE_TTL_SECONDS = 300
_REQUEST_TIMEOUT_SECONDS = 10
_REQUEST_MAX_ATTEMPTS = 3
_REQUEST_BACKOFF_SECONDS = 1.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
WIKI_HEADERS = {
    "User-Agent": "WatsonLite/1.0 (educational project; clavijodario@gmail.com)"
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NON_WORD_RE = re.compile(r"\W+")


def _normalize_passage_text(text: str) -> str:
    return " ".join(_NON_WORD_RE.sub(" ", text.lower()).split())


def _retry_delay_seconds(response: Any, attempt: int) -> float:  # noqa: ANN401
    retry_after: str | None = None
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        retry_after_value = headers.get("Retry-After")
        if retry_after_value is not None:
            retry_after = str(retry_after_value)
    if retry_after:
        try:
            return float(max(float(retry_after), 0.0))
        except ValueError:
            logger.debug("Ignoring invalid Retry-After header: %s", retry_after)
    return float(_REQUEST_BACKOFF_SECONDS * (2**attempt))


def _request_json(
    url: str,
    *,
    params: dict[str, Any],
    cache_namespace: str,
    context: str,
) -> dict[str, Any] | None:
    for attempt in range(_REQUEST_MAX_ATTEMPTS):
        try:
            response = requests.get(
                url,
                params=params,
                headers=WIKI_HEADERS,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as err:  # pragma: no cover - defensive network isolation
            if attempt == _REQUEST_MAX_ATTEMPTS - 1:
                logger.warning(
                    "%s request failed (%s): %s", context, cache_namespace, err
                )
                return None
            wait = _REQUEST_BACKOFF_SECONDS * (2**attempt)
            logger.warning(
                "%s request failed (%s), retrying in %.1fs: %s",
                context,
                cache_namespace,
                wait,
                err,
            )
            time.sleep(wait)
            continue

        status = int(getattr(response, "status_code", 200))
        if status in _RETRYABLE_STATUS_CODES:
            if attempt == _REQUEST_MAX_ATTEMPTS - 1:
                logger.warning(
                    "%s request failed (%s): HTTP %s",
                    context,
                    cache_namespace,
                    status,
                )
                return None
            wait = _retry_delay_seconds(response, attempt)
            logger.warning(
                "%s request rate-limited/transient failure (%s): HTTP %s; retrying in %.1fs",
                context,
                cache_namespace,
                status,
                wait,
            )
            time.sleep(wait)
            continue

        if status >= 400:
            logger.warning(
                "%s request failed (%s): HTTP %s", context, cache_namespace, status
            )
            return None

        try:
            payload = response.json()
        except Exception as err:  # pragma: no cover - defensive parsing guard
            logger.warning(
                "%s response parse failed (%s): %s", context, cache_namespace, err
            )
            return None

        if not isinstance(payload, dict):
            logger.warning(
                "%s response was not a JSON object (%s)", context, cache_namespace
            )
            return None
        return payload

    return None


def _chunk_text(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
        if current and current_words + len(sentence_words) > CHUNK_SIZE:
            chunk = " ".join(current).strip()
            if len(chunk.split()) >= MIN_CHUNK_WORDS:
                chunks.append(chunk)
            current = current[-CHUNK_OVERLAP_SENTENCES:]
            current_words = sum(len(item.split()) for item in current)
        current.append(sentence)
        current_words += len(sentence_words)

    if current:
        chunk = " ".join(current).strip()
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)

    if chunks:
        return chunks

    words = text.split()
    fallback_chunks = []
    # Keep overlap in the fallback path so extraction still sees context that
    # straddles a hard word-count boundary when sentence splitting is unavailable.
    for i in range(0, len(words), FALLBACK_CHUNK_STEP):
        chunk = " ".join(words[i : i + CHUNK_SIZE]).strip()
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            fallback_chunks.append(chunk)
    return fallback_chunks


def fetch_mediawiki_passages(
    query: str,
    *,
    top_k: int = WIKI_SEARCH_LIMIT,
    api_url: str,
    article_base_url: str,
    cache_namespace: str,
) -> list[Passage]:
    """Fetch and chunk passages from a MediaWiki API-backed dataset."""
    cache = get_cache()
    normalized_query = query.lower().strip()
    cache_key = f"{cache_namespace}:passages:{normalized_query}:top_k={top_k}"
    cached = cache.get_or_sentinel(cache_key)
    if not is_cache_miss(cached):
        logger.debug("Cache hit: %s", cache_key)
        return [Passage(**p) for p in cached]

    search_params: dict[str, Any] = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": top_k,
        "format": "json",
        "utf8": 1,
    }
    payload = _request_json(
        api_url,
        params=search_params,
        cache_namespace=cache_namespace,
        context="Dataset search",
    )
    if payload is None:
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []
    results = payload.get("query", {}).get("search", [])

    def _fetch_article(title: str) -> list[Passage]:
        extract_params: dict[str, Any] = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        payload = _request_json(
            api_url,
            params=extract_params,
            cache_namespace=cache_namespace,
            context=f"Dataset extract for '{title}'",
        )
        if payload is None:
            return []
        pages = payload.get("query", {}).get("pages", {})
        chunks: list[Passage] = []
        seen_chunks: set[str] = set()
        article_url = f"{article_base_url}/{title.replace(' ', '_')}"
        for page in pages.values():
            text = page.get("extract", "")
            if not text:
                continue
            for chunk in _chunk_text(text):
                dedup_key = _normalize_passage_text(chunk)
                if dedup_key in seen_chunks:
                    continue
                seen_chunks.add(dedup_key)
                chunks.append(
                    Passage(
                        text=chunk,
                        source=title,
                        url=article_url,
                    )
                )
        return chunks

    titles = [item["title"] for item in results]
    passages: list[Passage] = []
    if titles:
        with ThreadPoolExecutor(max_workers=min(len(titles), 5)) as executor:
            futures = {executor.submit(_fetch_article, t): t for t in titles}
            for future in as_completed(futures):
                passages.extend(future.result())

    cache.set(cache_key, [p.__dict__ for p in passages])
    logger.debug("Cache set: %s (%d passages)", cache_key, len(passages))
    return passages


def fetch_page_by_title(  # pylint: disable=too-many-arguments
    title: str,
    *,
    api_url: str,
    article_base_url: str,
    cache_namespace: str,
) -> list[Passage]:
    """Fetch and chunk a single Wikipedia/MediaWiki page by its exact title."""
    cache = get_cache()
    cache_key = f"{cache_namespace}:page:{title}"
    cached = cache.get_or_sentinel(cache_key)
    if not is_cache_miss(cached):
        logger.debug("Cache hit: %s", cache_key)
        return [Passage(**p) for p in cached]

    extract_params: dict[str, Any] = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    payload = _request_json(
        api_url,
        params=extract_params,
        cache_namespace=cache_namespace,
        context=f"Page fetch for '{title}'",
    )
    if payload is None:
        return []
    pages = payload.get("query", {}).get("pages", {})
    chunks: list[Passage] = []
    seen_chunks: set[str] = set()
    article_url = f"{article_base_url}/{title.replace(' ', '_')}"
    for page in pages.values():
        text = page.get("extract", "")
        if not text:
            continue
        for chunk in _chunk_text(text):
            dedup_key = _normalize_passage_text(chunk)
            if dedup_key in seen_chunks:
                continue
            seen_chunks.add(dedup_key)
            chunks.append(
                Passage(
                    text=chunk,
                    source=title,
                    url=article_url,
                )
            )

    cache.set(cache_key, [p.__dict__ for p in chunks])
    logger.debug("Cache set: %s (%d passages)", cache_key, len(chunks))
    return chunks


def fetch_wikipedia_page_by_title(title: str) -> list[Passage]:
    """Fetch and chunk a single Wikipedia page by its exact title."""
    return fetch_page_by_title(
        title,
        api_url=WIKI_API,
        article_base_url="https://en.wikipedia.org/wiki",
        cache_namespace="wiki",
    )


def fetch_wikipedia_passages(
    query: str, *, top_k: int = WIKI_SEARCH_LIMIT
) -> list[Passage]:
    """Fetch passages from Wikipedia using the generic MediaWiki fetcher."""
    return fetch_mediawiki_passages(
        query,
        top_k=top_k,
        api_url=WIKI_API,
        article_base_url="https://en.wikipedia.org/wiki",
        cache_namespace="wiki",
    )


def fetch_wikibooks_passages(
    query: str, *, top_k: int = WIKI_SEARCH_LIMIT
) -> list[Passage]:
    """Fetch passages from Wikibooks using the generic MediaWiki fetcher."""
    return fetch_mediawiki_passages(
        query,
        top_k=top_k,
        api_url=WIKIBOOKS_API,
        article_base_url="https://en.wikibooks.org/wiki",
        cache_namespace="wikibooks",
    )


def _get_elasticsearch_setting_or_env(value: str | None, env_name: str) -> str:
    """Return an explicit setting, or fall back to the corresponding env var."""
    if value is not None:
        return value.strip()
    return os.getenv(env_name, "").strip()


def _elasticsearch_headers() -> dict[str, str]:
    """Build Elasticsearch HTTP headers, adding ApiKey auth when configured."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.getenv(ELASTICSEARCH_API_KEY_ENV, "").strip()
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    return headers


def _extract_text_from_hit(source_payload: dict[str, Any]) -> str:
    """Return first non-empty text from: text, content, body, passage, snippet."""
    text = (
        source_payload.get("text")
        or source_payload.get("content")
        or source_payload.get("body")
        or source_payload.get("passage")
        or source_payload.get("snippet")
        or ""
    )
    if not isinstance(text, str):
        return ""
    return text.strip()


def _to_passage(
    hit: dict[str, Any], source_payload: dict[str, Any], resolved_index: str, text: str
) -> Passage:
    """Map an Elasticsearch hit to Passage using title/source/name and url/source_url."""
    source = (
        source_payload.get("title")
        or source_payload.get("source")
        or source_payload.get("name")
        or f"{resolved_index}:{hit.get('_id', 'unknown')}"
    )
    url = source_payload.get("url") or source_payload.get("source_url") or ""
    if not isinstance(source, str):
        source = str(source)
    if not isinstance(url, str):
        url = str(url)
    return Passage(text=text, source=source, url=url)


def _huggingface_headers(token: str) -> dict[str, str]:
    """Build Hugging Face headers, adding Bearer auth when configured."""
    headers: dict[str, str] = {"Accept": "application/json"}
    stripped = token.strip()
    if stripped:
        headers["Authorization"] = f"Bearer {stripped}"
    return headers


def _extract_huggingface_row_text(row_payload: dict[str, Any]) -> str:
    """Extract passage text from a Hugging Face row payload."""
    direct_text = (
        row_payload.get("text")
        or row_payload.get("content")
        or row_payload.get("body")
        or row_payload.get("passage")
        or row_payload.get("snippet")
        or row_payload.get("document")
    )
    if isinstance(direct_text, str):
        return direct_text.strip()

    for value in row_payload.values():
        if isinstance(value, str) and len(value.split()) >= MIN_CHUNK_WORDS:
            return value.strip()
    return ""


def fetch_huggingface_passages(
    query: str,
    *,
    top_k: int = WIKI_SEARCH_LIMIT,
    dataset: str | None = None,
    config: str | None = None,
    split: str | None = None,
    token: str | None = None,
) -> list[Passage]:
    """Fetch passages from a Hugging Face dataset via datasets-server search."""
    resolved_dataset = _get_elasticsearch_setting_or_env(
        dataset, HUGGINGFACE_DATASET_ENV
    )
    resolved_config = _get_elasticsearch_setting_or_env(config, HUGGINGFACE_CONFIG_ENV)
    resolved_split = _get_elasticsearch_setting_or_env(split, HUGGINGFACE_SPLIT_ENV)
    resolved_token = _get_elasticsearch_setting_or_env(token, HUGGINGFACE_TOKEN_ENV)

    if not resolved_dataset or not resolved_split:
        logger.warning(
            "Hugging Face dataset requested but not configured."
            " Set %s and %s (or pass config values).",
            HUGGINGFACE_DATASET_ENV,
            HUGGINGFACE_SPLIT_ENV,
        )
        return []

    cache = get_cache()
    normalized_query = query.lower().strip()
    cache_key = (
        "hf:passages:"
        f"{resolved_dataset}:{resolved_config}:{resolved_split}:{normalized_query}:top_k={top_k}"
    )
    cached = cache.get_or_sentinel(cache_key)
    if not is_cache_miss(cached):
        logger.debug("Cache hit: %s", cache_key)
        return [Passage(**p) for p in cached]

    params: dict[str, Any] = {
        "dataset": resolved_dataset,
        "split": resolved_split,
        "q": query,
        "limit": top_k,
    }
    if resolved_config:
        params["config"] = resolved_config

    try:
        response = requests.get(
            HUGGINGFACE_DATASET_SERVER_SEARCH,
            params=params,
            headers=_huggingface_headers(resolved_token),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as err:  # pragma: no cover - defensive network isolation
        logger.warning("Hugging Face datasets-server request failed: %s", err)
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    if response.status_code >= 400:
        logger.warning(
            "Hugging Face datasets-server request failed: HTTP %s",
            int(response.status_code),
        )
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    try:
        data = response.json()
    except Exception as err:  # pragma: no cover - defensive parsing guard
        logger.warning("Hugging Face datasets-server response parse failed: %s", err)
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    if not isinstance(data, dict):
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list):
        raw_rows = data.get("hits")
    if not isinstance(raw_rows, list):
        raw_rows = data.get("results")
    if not isinstance(raw_rows, list):
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    dataset_url = f"https://huggingface.co/datasets/{resolved_dataset}"
    passages: list[Passage] = []
    seen_chunks: set[str] = set()
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        row_payload = item.get("row")
        if not isinstance(row_payload, dict):
            row_payload = item
        text = _extract_huggingface_row_text(row_payload)
        if not text:
            continue
        dedup_key = _normalize_passage_text(text)
        if dedup_key in seen_chunks:
            continue
        seen_chunks.add(dedup_key)
        source = (
            row_payload.get("title")
            or row_payload.get("source")
            or row_payload.get("name")
            or resolved_dataset
        )
        if not isinstance(source, str):
            source = str(source)
        passages.append(
            Passage(
                text=text,
                source=source,
                url=dataset_url,
            )
        )

    cache.set(cache_key, [p.__dict__ for p in passages])
    logger.debug("Cache set: %s (%d passages)", cache_key, len(passages))
    return passages


def fetch_elasticsearch_passages(
    query: str,
    *,
    top_k: int = WIKI_SEARCH_LIMIT,
    base_url: str | None = None,
    index: str | None = None,
) -> list[Passage]:
    """Fetch passages from an Elasticsearch index."""
    resolved_base_url = _get_elasticsearch_setting_or_env(
        base_url, ELASTICSEARCH_URL_ENV
    )
    resolved_index = _get_elasticsearch_setting_or_env(
        index, ELASTICSEARCH_INDEX_ENV
    )
    if not resolved_base_url or not resolved_index:
        logger.warning(
            "Elasticsearch dataset requested but not configured."
            " Set %s and %s (or pass config values).",
            ELASTICSEARCH_URL_ENV,
            ELASTICSEARCH_INDEX_ENV,
        )
        return []

    cache = get_cache()
    normalized_query = query.lower().strip()
    cache_key = (
        "elastic:passages:"
        f"{resolved_index}:{normalized_query}:top_k={top_k}:base={resolved_base_url}"
    )
    cached = cache.get_or_sentinel(cache_key)
    if not is_cache_miss(cached):
        logger.debug("Cache hit: %s", cache_key)
        return [Passage(**p) for p in cached]

    endpoint = f"{resolved_base_url.rstrip('/')}/{resolved_index}/_search"
    payload: dict[str, Any] = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "text", "content", "body", "passage", "snippet"],
                "type": "best_fields",
            }
        },
    }
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=_elasticsearch_headers(),
            timeout=ELASTICSEARCH_TIMEOUT_SECONDS,
        )
    except Exception as err:  # pragma: no cover - defensive network isolation
        logger.warning("Elasticsearch request failed: %s", err)
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    if response.status_code >= 400:
        logger.warning(
            "Elasticsearch request failed: HTTP %s", int(response.status_code)
        )
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    try:
        data = response.json()
    except Exception as err:  # pragma: no cover - defensive parsing guard
        logger.warning("Elasticsearch response parse failed: %s", err)
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    hits = data.get("hits", {}).get("hits", [])
    if not isinstance(hits, list):
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    passages: list[Passage] = []
    seen_chunks: set[str] = set()
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source_payload = hit.get("_source")
        if not isinstance(source_payload, dict):
            continue
        text = _extract_text_from_hit(source_payload)
        if not text:
            continue
        dedup_key = _normalize_passage_text(text)
        if dedup_key in seen_chunks:
            continue
        seen_chunks.add(dedup_key)
        passages.append(_to_passage(hit, source_payload, resolved_index, text))

    cache.set(cache_key, [p.__dict__ for p in passages])
    logger.debug("Cache set: %s (%d passages)", cache_key, len(passages))
    return passages


class BM25Retriever:
    def __init__(self) -> None:
        self.passages: list[Passage] = []
        self.retriever = None

    def index(self, passages: list[Passage]) -> None:
        self.passages = passages
        corpus = [p.text for p in passages]
        tokenized = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(tokenized)
        self.retriever = retriever
        logger.debug("Indexed %d passages", len(passages))

    def retrieve(self, query: str, top_k: int = 10) -> list[Passage]:
        if not self.retriever or not self.passages:
            return []

        tokenized_query = bm25s.tokenize([query], stopwords="en")
        docs, scores = self.retriever.retrieve(
            tokenized_query, k=min(top_k, len(self.passages))
        )

        text_to_passage = {p.text: p for p in self.passages}
        retrieved = []
        for doc_text, score in zip(docs[0], scores[0]):
            p = text_to_passage.get(doc_text)
            if p:
                # Copy so that concurrent retrievers don't overwrite each
                # other's score/rank on the shared Passage object.
                p = copy.copy(p)
                p.score = float(score)
                retrieved.append(p)

        for rank, p in enumerate(retrieved):
            p.rank = rank + 1

        return retrieved

    def fetch_and_retrieve(self, query: str, top_k: int = 10) -> list[Passage]:
        logger.debug("Fetching Wikipedia for: '%s'", query)
        passages = fetch_wikipedia_passages(query)
        if not passages:
            return []
        self.index(passages)
        return self.retrieve(query, top_k=top_k)
