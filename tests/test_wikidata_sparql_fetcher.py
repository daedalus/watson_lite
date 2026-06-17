from unittest.mock import MagicMock, patch

from watson_lite.core.cache import SENTINEL
from watson_lite.core.models import Passage
from watson_lite.retrieval.wikidata_sparql_fetcher import (
    fetch_wikidata_sparql_passages,
)


class TestFetchWikidataSparqlPassages:
    def setup_method(self) -> None:
        self.cache_patcher = patch(
            "watson_lite.retrieval.wikidata_sparql_fetcher.get_cache"
        )
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache
        self.sleep_patcher = patch("watson_lite.core.network.time.sleep")
        self.mock_sleep = self.sleep_patcher.start()

    def teardown_method(self) -> None:
        self.cache_patcher.stop()
        self.sleep_patcher.stop()

    @patch("watson_lite.core.network.requests.get")
    def test_sparql_success(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/Q42",
                        },
                        "itemLabel": {
                            "type": "literal",
                            "xml:lang": "en",
                            "value": "Douglas Adams",
                        },
                        "description": {
                            "type": "literal",
                            "xml:lang": "en",
                            "value": "English author",
                        },
                    }
                ]
            }
        }
        mock_get.return_value = response

        result = fetch_wikidata_sparql_passages("douglas adams")

        assert len(result) == 1
        assert result[0].source == "Wikidata"
        assert "Douglas Adams" in result[0].text
        assert "English author" in result[0].text
        assert result[0].url == "http://www.wikidata.org/entity/Q42"

    @patch("watson_lite.core.network.requests.get")
    def test_sparql_empty_bindings(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"results": {"bindings": []}}
        mock_get.return_value = response

        result = fetch_wikidata_sparql_passages("nonexistentxyz")

        assert result == []

    @patch("watson_lite.core.network.requests.get")
    def test_sparql_http_error(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 500
        mock_get.return_value = response

        result = fetch_wikidata_sparql_passages("python")

        assert result == []

    @patch("watson_lite.core.network.requests.get")
    def test_sparql_retries_on_429(self, mock_get: MagicMock) -> None:
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}
        ok = MagicMock()
        ok.status_code = 200
        ok.json.return_value = {"results": {"bindings": []}}
        mock_get.side_effect = [rate_limited, ok]

        result = fetch_wikidata_sparql_passages("python")

        assert result == []
        assert mock_get.call_count == 2
        self.mock_sleep.assert_called_once()

    @patch("watson_lite.core.network.requests.get")
    def test_sparql_custom_endpoint(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"results": {"bindings": []}}
        mock_get.return_value = response

        fetch_wikidata_sparql_passages(
            "python", endpoint="https://custom.example/sparql"
        )

        assert mock_get.call_args[0][0] == "https://custom.example/sparql"

    @patch("watson_lite.core.network.requests.get")
    def test_sparql_top_k_capped(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"results": {"bindings": []}}
        mock_get.return_value = response

        fetch_wikidata_sparql_passages("python", top_k=200)

        sparql_sent = mock_get.call_args[1]["params"]["query"]
        assert "LIMIT 50" in sparql_sent

    def test_sparql_returns_cached_result(self) -> None:
        cached_passage = Passage(
            text="cached text", source="Wikidata", url="https://wikidata.org"
        )
        self.mock_cache.get_or_sentinel.return_value = [cached_passage.__dict__]

        result = fetch_wikidata_sparql_passages("python")

        assert len(result) == 1
        assert result[0].text == "cached text"
