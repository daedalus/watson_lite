from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol

from watson_lite.core.models import Passage

logger = logging.getLogger(__name__)


class DatasetProvider(Protocol):
    """Queryable dataset provider used by the dataset query engine."""

    name: str

    def fetch_passages(self, query: str, top_k: int) -> list[Passage]:
        """Return passages for a query from one dataset."""


@dataclass(frozen=True)
class FunctionDatasetProvider:
    """Adapter that turns a function into a dataset provider."""

    name: str
    fetcher: Callable[[str, int], list[Passage]]

    def fetch_passages(self, query: str, top_k: int) -> list[Passage]:
        return self.fetcher(query, top_k)


class DatasetQueryEngine:
    """Aggregates queryable passages from modular dataset providers."""

    def __init__(
        self,
        providers: tuple[DatasetProvider, ...],
        enabled_datasets: tuple[str, ...],
    ) -> None:
        provider_map = {provider.name: provider for provider in providers}
        self._providers = provider_map
        self._enabled_datasets = enabled_datasets

    def query(self, query: str, top_k: int) -> list[Passage]:
        passages: list[Passage] = []
        for dataset_name in self._enabled_datasets:
            provider = self._providers.get(dataset_name)
            if provider is None:
                logger.warning("Unknown dataset configured: '%s'", dataset_name)
                continue
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
