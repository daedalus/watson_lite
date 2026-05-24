import copy
import logging
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
CHUNK_SIZE = 200
_NEGATIVE_CACHE_TTL_SECONDS = 300
WIKI_HEADERS = {
    "User-Agent": "WatsonLite/1.0 (educational project; clavijodario@gmail.com)"
}


def fetch_mediawiki_passages(
    query: str,
    *,
    top_k: int,
    api_url: str,
    article_base_url: str,
    cache_namespace: str,
) -> list[Passage]:
    cache = get_cache()
    cache_key = f"{cache_namespace}:passages:{query.lower().strip()}"
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
    try:
        resp = requests.get(
            api_url, params=search_params, headers=WIKI_HEADERS, timeout=10
        )
        results = resp.json().get("query", {}).get("search", [])
    except Exception as e:
        logger.warning("Dataset search error (%s): %s", cache_namespace, e)
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    def _fetch_article(title: str) -> list[Passage]:
        extract_params: dict[str, Any] = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        try:
            eresp = requests.get(
                api_url, params=extract_params, headers=WIKI_HEADERS, timeout=10
            )
            pages = eresp.json().get("query", {}).get("pages", {})
            chunks: list[Passage] = []
            for page in pages.values():
                text = page.get("extract", "")
                if not text:
                    continue
                words = text.split()
                for i in range(0, len(words), CHUNK_SIZE // 2):
                    chunk = " ".join(words[i : i + CHUNK_SIZE])
                    if len(chunk.split()) < 20:
                        continue
                    chunks.append(
                        Passage(
                            text=chunk,
                            source=title,
                            url=(
                                f"{article_base_url}/{title.replace(' ', '_')}"
                            ),
                        )
                    )
            return chunks
        except Exception as e:
            logger.warning("Extract error for '%s': %s", title, e)
            return []

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
