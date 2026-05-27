import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from watson_lite.core.config import FeatureConfig
from watson_lite.core.models import Passage
from watson_lite.retrieval.dataset_plugins import (
    DatasetRetrieverPlugin,
    build_dataset_plugin_registry,
)
from watson_lite.retrieval.offline_dataset_retriever import (
    fetch_offline_dataset_passages,
)


class TestDatasetPlugins:
    def test_builtin_registry_contains_online_and_offline_plugins(self) -> None:
        registry = build_dataset_plugin_registry(FeatureConfig.baseline())

        online_names = {plugin.name for plugin in registry.list(mode="online")}
        offline_names = {plugin.name for plugin in registry.list(mode="offline")}

        assert "wikipedia" in online_names
        assert "wikipedia_offline" in offline_names
        for online_name in online_names:
            assert f"{online_name}_offline" in offline_names

    def test_offline_retriever_uses_local_corpus(self, tmp_path: Path) -> None:
        corpus_path = tmp_path / "wikipedia.jsonl"
        corpus_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "text": "Paris is the capital of France.",
                            "source": "Wikipedia",
                            "url": "https://example.org/paris",
                        }
                    ),
                    json.dumps(
                        {
                            "text": "Madrid is the capital of Spain.",
                            "source": "Wikipedia",
                            "url": "https://example.org/madrid",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        passages = fetch_offline_dataset_passages(
            "capital france",
            top_k=1,
            dataset_name="wikipedia",
            base_dir=str(tmp_path),
        )

        assert len(passages) == 1
        assert passages[0].text == "Paris is the capital of France."
        assert passages[0].rank == 1
        assert passages[0].score > 0

    def test_registry_loads_entrypoint_plugins(self) -> None:
        @dataclass
        class _FakeEntryPoint:
            name: str

            def load(self) -> object:
                plugin = DatasetRetrieverPlugin(
                    name="custom_online",
                    mode="online",
                    description="custom plugin",
                    fetcher=lambda query, *, top_k: [
                        Passage(text=query, source="custom", url="", rank=top_k)
                    ],
                )
                return lambda: (plugin,)

        class _FakeEntryPoints:
            def select(self, *, group: str) -> list[_FakeEntryPoint]:
                if group == "watson_lite.dataset_retrievers":
                    return [_FakeEntryPoint(name="custom")]
                return []

        with patch(
            "watson_lite.retrieval.dataset_plugins.entry_points",
            return_value=_FakeEntryPoints(),
        ):
            registry = build_dataset_plugin_registry(FeatureConfig.baseline())

        plugin = registry.get("custom_online")
        assert plugin is not None
        assert plugin.mode == "online"
        assert plugin.source == "entrypoint:custom"
