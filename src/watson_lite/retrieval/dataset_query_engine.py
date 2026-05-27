from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from watson_lite.core.models import Passage

logger = logging.getLogger(__name__)


class PassageFetcher(Protocol):
    """Callable used by dataset providers."""

    def __call__(self, query: str, *, top_k: int) -> list[Passage]:
        """Fetch passages from one dataset."""


@dataclass(frozen=True)
class DatasetProvider:
    """Dataset provider adapter with a stable name and fetch implementation."""

    name: str
    fetcher: PassageFetcher

    def fetch_passages(self, query: str, top_k: int) -> list[Passage]:
        return self.fetcher(query, top_k=top_k)


class DatasetQueryEngine:
    """Aggregates queryable passages from modular dataset providers.

    Providers are queried concurrently via a ThreadPoolExecutor so that
    enabling multiple datasets (e.g. wikipedia + pubmed + arxiv) does not
    sequence their HTTP requests.
    """

    def __init__(
        self,
        providers: tuple[DatasetProvider, ...],
        enabled_datasets: tuple[str, ...],
    ) -> None:
        provider_map = {provider.name: provider for provider in providers}
        self._providers = provider_map
        self._enabled_datasets = enabled_datasets
        self._query_cache: dict[str, list[Passage]] = {}

    def query(self, query: str, top_k: int) -> list[Passage]:
        passages: list[Passage] = []
        uncached: list[tuple[str, DatasetProvider]] = []

        for dataset_name in self._enabled_datasets:
            cache_key = f"dq:{dataset_name}:{query}:{top_k}"
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                passages.extend(cached)
                continue

            provider = self._providers.get(dataset_name)
            if provider is None:
                logger.warning("Unknown dataset configured: '%s'", dataset_name)
                continue
            uncached.append((dataset_name, provider))

        if uncached:
            with ThreadPoolExecutor(max_workers=len(uncached)) as executor:
                future_map = {
                    executor.submit(provider.fetch_passages, query, top_k): name
                    for name, provider in uncached
                }
                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        fetched = future.result()
                        cache_key = f"dq:{name}:{query}:{top_k}"
                        self._query_cache[cache_key] = fetched
                        passages.extend(fetched)
                    except Exception as err:  # pragma: no cover - defensive isolation
                        logger.warning(
                            "Dataset provider '%s' failed for query '%s': %s",
                            name,
                            query,
                            err,
                        )

        return passages
