from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Literal, Protocol

from watson_lite.retrieval.bm25_retriever import (
    fetch_arxiv_passages,
    fetch_dbpedia_passages,
    fetch_dbpedia_sparql_passages,
    fetch_elasticsearch_passages,
    fetch_huggingface_passages,
    fetch_oeis_passages,
    fetch_openlibrary_passages,
    fetch_pubmed_passages,
    fetch_stackexchange_passages,
    fetch_wikibooks_passages,
    fetch_wikinews_passages,
    fetch_wikipedia_passages,
    fetch_wikiquote_passages,
    fetch_wikisource_passages,
)
from watson_lite.retrieval.dataset_query_engine import DatasetProvider
from watson_lite.retrieval.offline_dataset_retriever import (
    fetch_offline_dataset_passages,
)

if TYPE_CHECKING:
    from watson_lite.core.config import FeatureConfig
    from watson_lite.core.models import Passage

logger = logging.getLogger(__name__)
PluginMode = Literal["online", "offline"]
_PLUGIN_ENTRYPOINT_GROUP = "watson_lite.dataset_retrievers"


class PassageFetcher(Protocol):
    """Callable used by dataset retriever plugins."""

    def __call__(self, query: str, *, top_k: int) -> list[Passage]:
        """Fetch passages from one dataset plugin."""


@dataclass(frozen=True)
class DatasetRetrieverPlugin:
    """Pluggable dataset retriever."""

    name: str
    mode: PluginMode
    description: str
    fetcher: PassageFetcher
    source: str = "builtin"

    def to_provider(self) -> DatasetProvider:
        return DatasetProvider(name=self.name, fetcher=self.fetcher)


class DatasetPluginRegistry:
    """Registry for dataset retriever plugins."""

    def __init__(self, plugins: tuple[DatasetRetrieverPlugin, ...]) -> None:
        self._plugins = {plugin.name: plugin for plugin in plugins}

    def get(self, name: str) -> DatasetRetrieverPlugin | None:
        return self._plugins.get(name)

    def list(self, mode: PluginMode | None = None) -> tuple[DatasetRetrieverPlugin, ...]:
        plugins = tuple(self._plugins.values())
        if mode is None:
            return tuple(sorted(plugins, key=lambda plugin: plugin.name))
        return tuple(
            sorted(
                (plugin for plugin in plugins if plugin.mode == mode),
                key=lambda plugin: plugin.name,
            )
        )

    def provider_tuple(self) -> tuple[DatasetProvider, ...]:
        return tuple(plugin.to_provider() for plugin in self.list())

    def has_all(self, plugin_names: tuple[str, ...]) -> bool:
        return all(name in self._plugins for name in plugin_names)

    def missing(self, plugin_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name in plugin_names if name not in self._plugins)


def _builtin_online_plugins(config: FeatureConfig) -> tuple[DatasetRetrieverPlugin, ...]:
    return (
        DatasetRetrieverPlugin(
            "wikipedia",
            mode="online",
            description="Wikipedia REST retriever",
            fetcher=fetch_wikipedia_passages,
        ),
        DatasetRetrieverPlugin(
            "wikibooks",
            mode="online",
            description="Wikibooks REST retriever",
            fetcher=fetch_wikibooks_passages,
        ),
        DatasetRetrieverPlugin(
            "wikiquote",
            mode="online",
            description="Wikiquote REST retriever",
            fetcher=fetch_wikiquote_passages,
        ),
        DatasetRetrieverPlugin(
            "wikisource",
            mode="online",
            description="Wikisource REST retriever",
            fetcher=fetch_wikisource_passages,
        ),
        DatasetRetrieverPlugin(
            "wikinews",
            mode="online",
            description="Wikinews REST retriever",
            fetcher=fetch_wikinews_passages,
        ),
        DatasetRetrieverPlugin(
            "pubmed",
            mode="online",
            description="PubMed E-utilities retriever",
            fetcher=fetch_pubmed_passages,
        ),
        DatasetRetrieverPlugin(
            "arxiv",
            mode="online",
            description="arXiv API retriever",
            fetcher=fetch_arxiv_passages,
        ),
        DatasetRetrieverPlugin(
            "openlibrary",
            mode="online",
            description="OpenLibrary API retriever",
            fetcher=fetch_openlibrary_passages,
        ),
        DatasetRetrieverPlugin(
            "stackexchange",
            mode="online",
            description="Stack Exchange API retriever",
            fetcher=fetch_stackexchange_passages,
        ),
        DatasetRetrieverPlugin(
            "dbpedia",
            mode="online",
            description="DBpedia lookup retriever",
            fetcher=fetch_dbpedia_passages,
        ),
        DatasetRetrieverPlugin(
            "dbpedia_sparql",
            mode="online",
            description="DBpedia SPARQL retriever",
            fetcher=fetch_dbpedia_sparql_passages,
        ),
        DatasetRetrieverPlugin(
            "oeis",
            mode="online",
            description="OEIS retriever",
            fetcher=fetch_oeis_passages,
        ),
        DatasetRetrieverPlugin(
            "elasticsearch",
            mode="online",
            description="Elasticsearch index retriever",
            fetcher=lambda query, *, top_k: fetch_elasticsearch_passages(
                query,
                top_k=top_k,
                base_url=config.elasticsearch_url,
                index=config.elasticsearch_index,
            ),
        ),
        DatasetRetrieverPlugin(
            "huggingface",
            mode="online",
            description="Hugging Face datasets-server retriever",
            fetcher=lambda query, *, top_k: fetch_huggingface_passages(
                query,
                top_k=top_k,
                dataset=config.huggingface_dataset,
                config=config.huggingface_config,
                split=config.huggingface_split,
                token=config.huggingface_token,
            ),
        ),
    )


def _builtin_offline_plugins(config: FeatureConfig) -> tuple[DatasetRetrieverPlugin, ...]:
    plugins: list[DatasetRetrieverPlugin] = []
    for online_plugin in _builtin_online_plugins(config):
        dataset_name = online_plugin.name
        plugins.append(
            DatasetRetrieverPlugin(
                f"{dataset_name}_offline",
                mode="offline",
                description=(
                    f"Offline local retriever for '{dataset_name}' "
                    "(JSON/JSONL corpus file)"
                ),
                fetcher=partial(
                    fetch_offline_dataset_passages,
                    dataset_name=dataset_name,
                    base_dir=config.offline_dataset_dir,
                ),
            )
        )
    return tuple(plugins)


def _coerce_external_plugins(
    loaded: object,
    *,
    source: str,
) -> tuple[DatasetRetrieverPlugin, ...]:
    if isinstance(loaded, DatasetRetrieverPlugin):
        return (DatasetRetrieverPlugin(**{**loaded.__dict__, "source": source}),)
    if isinstance(loaded, Iterable) and not isinstance(loaded, (str, bytes)):
        normalized: list[DatasetRetrieverPlugin] = []
        for item in loaded:
            if not isinstance(item, DatasetRetrieverPlugin):
                continue
            normalized.append(DatasetRetrieverPlugin(**{**item.__dict__, "source": source}))
        return tuple(normalized)
    return ()


def _load_external_plugins() -> tuple[DatasetRetrieverPlugin, ...]:
    loaded_plugins: list[DatasetRetrieverPlugin] = []
    entry_point_collection = entry_points()
    if hasattr(entry_point_collection, "select"):
        selected = tuple(entry_point_collection.select(group=_PLUGIN_ENTRYPOINT_GROUP))
    else:
        selected = ()
    for plugin_entry_point in selected:
        try:
            loaded = plugin_entry_point.load()
            value = loaded() if callable(loaded) else loaded
        except (AttributeError, ImportError, TypeError) as err:
            logger.warning(
                "Failed loading dataset plugin entry point '%s': %s",
                plugin_entry_point.name,
                err,
            )
            continue
        source = f"entrypoint:{plugin_entry_point.name}"
        loaded_plugins.extend(_coerce_external_plugins(value, source=source))
    return tuple(loaded_plugins)


def build_dataset_plugin_registry(config: FeatureConfig) -> DatasetPluginRegistry:
    """Build the dataset plugin registry with built-ins and external entry points."""
    plugins = (
        *_builtin_online_plugins(config),
        *_builtin_offline_plugins(config),
        *_load_external_plugins(),
    )
    return DatasetPluginRegistry(tuple(plugins))
