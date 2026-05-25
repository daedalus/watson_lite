import copy
import logging
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
                logger.warning("%s request failed (%s): %s", context, cache_namespace, err)
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
            logger.warning("%s request failed (%s): HTTP %s", context, cache_namespace, status)
            return None

        try:
            payload = response.json()
        except Exception as err:  # pragma: no cover - defensive parsing guard
            logger.warning("%s response parse failed (%s): %s", context, cache_namespace, err)
            return None

        if not isinstance(payload, dict):
            logger.warning("%s response was not a JSON object (%s)", context, cache_namespace)
            return None
        return payload

    return None


def _chunk_text(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
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
