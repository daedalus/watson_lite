from __future__ import annotations

import logging
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
    """Aggregates queryable passages from modular dataset providers."""

    def __init__(
        self,
        providers: tuple[DatasetProvider, ...],
        enabled_datasets: tuple[str, ...],
    ) -> None:
        provider_map = {provider.name: provider for provider in providers}
        self._providers = provider_map
        unknown_datasets = tuple(
            dataset_name
            for dataset_name in enabled_datasets
            if dataset_name not in provider_map
        )
        for dataset_name in unknown_datasets:
            logger.warning("Unknown dataset configured: '%s'", dataset_name)
        self._enabled_datasets = tuple(
            dataset_name
            for dataset_name in enabled_datasets
            if dataset_name in provider_map
        )

    def query(self, query: str, top_k: int) -> list[Passage]:
        passages: list[Passage] = []
        for dataset_name in self._enabled_datasets:
            provider = self._providers[dataset_name]
            try:
                passages.extend(provider.fetch_passages(query, top_k))
            except Exception as err:  # pragma: no cover - defensive isolation
                logger.warning(
                    "Dataset provider '%s' failed for query '%s': %s",
                    dataset_name,
                    query,
                    err,
                )
        return passages
