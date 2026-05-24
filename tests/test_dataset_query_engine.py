from unittest.mock import MagicMock

from watson_lite.core.models import Passage
from watson_lite.retrieval.dataset_query_engine import (
    DatasetQueryEngine,
    FunctionDatasetProvider,
)


class TestDatasetQueryEngine:
    def test_aggregates_multiple_enabled_datasets(self) -> None:
        wiki_fetcher = MagicMock(
            return_value=[
                Passage(text="wiki text", source="Wikipedia", url="https://wiki")
            ]
        )
        wikibooks_fetcher = MagicMock(
            return_value=[
                Passage(
                    text="wikibooks text",
                    source="Wikibooks",
                    url="https://wikibooks",
                )
            ]
        )
        engine = DatasetQueryEngine(
            providers=(
                FunctionDatasetProvider("wikipedia", wiki_fetcher),
                FunctionDatasetProvider("wikibooks", wikibooks_fetcher),
            ),
            enabled_datasets=("wikipedia", "wikibooks"),
        )

        result = engine.query("python", top_k=5)

        assert len(result) == 2
        wiki_fetcher.assert_called_once_with("python", 5)
        wikibooks_fetcher.assert_called_once_with("python", 5)

    def test_ignores_unknown_datasets(self) -> None:
        wiki_fetcher = MagicMock(return_value=[])
        engine = DatasetQueryEngine(
            providers=(FunctionDatasetProvider("wikipedia", wiki_fetcher),),
            enabled_datasets=("wikipedia", "unknown"),
        )

        result = engine.query("python", top_k=5)

        assert result == []
        wiki_fetcher.assert_called_once_with("python", 5)

    def test_isolates_failing_provider(self) -> None:
        broken_fetcher = MagicMock(side_effect=RuntimeError("boom"))
        wiki_fetcher = MagicMock(
            return_value=[
                Passage(text="ok", source="Wikipedia", url="https://wiki"),
            ]
        )
        engine = DatasetQueryEngine(
            providers=(
                FunctionDatasetProvider("broken", broken_fetcher),
                FunctionDatasetProvider("wikipedia", wiki_fetcher),
            ),
            enabled_datasets=("broken", "wikipedia"),
        )

        result = engine.query("python", top_k=5)

        assert len(result) == 1
        assert result[0].text == "ok"
        broken_fetcher.assert_called_once_with("python", 5)
        wiki_fetcher.assert_called_once_with("python", 5)
