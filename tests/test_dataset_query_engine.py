from unittest.mock import MagicMock

import pytest

from watson_lite.core.models import Passage
from watson_lite.retrieval.dataset_query_engine import (
    DatasetProvider,
    DatasetQueryEngine,
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
                DatasetProvider("wikipedia", wiki_fetcher),
                DatasetProvider("wikibooks", wikibooks_fetcher),
            ),
            enabled_datasets=("wikipedia", "wikibooks"),
        )

        result = engine.query("python", top_k=5)

        assert len(result) == 2
        wiki_fetcher.assert_called_once_with("python", top_k=5)
        wikibooks_fetcher.assert_called_once_with("python", top_k=5)

    def test_ignores_unknown_datasets(self) -> None:
        wiki_fetcher = MagicMock(return_value=[])
        engine = DatasetQueryEngine(
            providers=(DatasetProvider("wikipedia", wiki_fetcher),),
            enabled_datasets=("wikipedia", "unknown"),
        )

        result = engine.query("python", top_k=5)

        assert result == []
        wiki_fetcher.assert_called_once_with("python", top_k=5)

    def test_logs_unknown_datasets_once_at_initialization(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wiki_fetcher = MagicMock(return_value=[])
        with caplog.at_level("WARNING"):
            engine = DatasetQueryEngine(
                providers=(DatasetProvider("wikipedia", wiki_fetcher),),
                enabled_datasets=("unknown", "wikipedia"),
            )

            engine.query("python", top_k=5)
            engine.query("python", top_k=5)

        warnings = [
            record.message
            for record in caplog.records
            if "Unknown dataset configured" in record.message
        ]
        assert warnings == ["Unknown dataset configured: 'unknown'"]

    def test_continues_querying_after_provider_failure(self) -> None:
        broken_fetcher = MagicMock(side_effect=RuntimeError("boom"))
        wiki_fetcher = MagicMock(
            return_value=[
                Passage(text="ok", source="Wikipedia", url="https://wiki"),
            ]
        )
        engine = DatasetQueryEngine(
            providers=(
                DatasetProvider("broken", broken_fetcher),
                DatasetProvider("wikipedia", wiki_fetcher),
            ),
            enabled_datasets=("broken", "wikipedia"),
        )

        result = engine.query("python", top_k=5)

        assert len(result) == 1
        assert result[0].text == "ok"
        broken_fetcher.assert_called_once_with("python", top_k=5)
        wiki_fetcher.assert_called_once_with("python", top_k=5)
