import pytest
from unittest.mock import MagicMock, patch

from watson_lite.core.cache import SENTINEL
from watson_lite.core.models import EntityFact, GraphResult
from watson_lite.graph.wikidata import WikidataGraph


class TestWikidataGraph:
    def setup_method(self) -> None:
        self.sparql_patcher = patch("watson_lite.graph.wikidata.SPARQLWrapper")
        self.mock_sparql_cls = self.sparql_patcher.start()
        self.mock_sparql = MagicMock()
        self.mock_sparql_cls.return_value = self.mock_sparql

        self.cache_patcher = patch("watson_lite.graph.wikidata.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        # Default: every key is a cache miss.
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache

        self.graph = WikidataGraph()

    def teardown_method(self) -> None:
        self.sparql_patcher.stop()
        self.cache_patcher.stop()

    def test_constructor_configures_sparql(self) -> None:
        self.mock_sparql_cls.assert_called_once_with(
            "https://query.wikidata.org/sparql"
        )
        self.mock_sparql.addCustomHttpHeader.assert_called_once_with(
            "User-Agent", "WatsonLite/1.0 (research project; python)"
        )
        self.mock_sparql.setReturnFormat.assert_called_once()

    def test_find_entity_id_cache_hit(self) -> None:
        self.mock_cache.get_or_sentinel.return_value = "Q243"
        qid = self.graph.find_entity_id("Eiffel Tower")
        assert qid == "Q243"

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_find_entity_id_api_success(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"search": [{"id": "Q243"}]}
        mock_get.return_value = mock_resp

        qid = self.graph.find_entity_id("Eiffel Tower")
        assert qid == "Q243"
        self.mock_cache.set.assert_called_once_with("wd:entity:eiffel tower", "Q243")

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_find_entity_id_429_fallback_sparql_success(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.return_value.status_code = 429
        self.mock_sparql.query.return_value.convert.return_value = {
            "results": {
                "bindings": [{"item": {"value": "http://www.wikidata.org/entity/Q243"}}]
            }
        }

        qid = self.graph.find_entity_id("Eiffel Tower")
        assert qid == "Q243"

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_find_entity_id_429_fallback_sparql_empty(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.return_value.status_code = 429
        self.mock_sparql.query.return_value.convert.return_value = {
            "results": {"bindings": []}
        }

        qid = self.graph.find_entity_id("Eiffel Tower")
        assert qid is None

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_find_entity_id_api_no_search_results(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"search": []}
        mock_get.return_value = mock_resp

        qid = self.graph.find_entity_id("Nonexistent")
        assert qid is None

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_find_entity_id_api_exception(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = Exception("Connection error")

        qid = self.graph.find_entity_id("Eiffel Tower")
        assert qid is None

    def test_find_entity_id_sparql_success(self) -> None:
        self.mock_sparql.query.return_value.convert.return_value = {
            "results": {
                "bindings": [{"item": {"value": "http://www.wikidata.org/entity/Q243"}}]
            }
        }

        qid = self.graph._find_entity_id_sparql("Eiffel Tower")
        assert qid == "Q243"

    def test_find_entity_id_sparql_empty(self) -> None:
        self.mock_sparql.query.return_value.convert.return_value = {
            "results": {"bindings": []}
        }

        qid = self.graph._find_entity_id_sparql("Nonexistent")
        assert qid is None

    def test_find_entity_id_sparql_escapes_quotes(self) -> None:
        """Entity names containing quotes must not break the SPARQL query."""
        self.mock_sparql.setQuery.reset_mock()
        self.mock_sparql.query.return_value.convert.return_value = {
            "results": {"bindings": []}
        }
        # Should not raise; the quote must be escaped in the generated query.
        qid = self.graph._find_entity_id_sparql('O"Brien')
        assert qid is None
        called_query: str = self.mock_sparql.setQuery.call_args[0][0]
        assert '\\"' in called_query

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_success(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "entities": {
                "Q243": {
                    "claims": {
                        "P31": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {
                                        "type": "string",
                                        "value": "val1",
                                    },
                                }
                            }
                        ]
                    }
                }
            }
        }
        mock_get.return_value = mock_resp

        facts = self.graph.get_entity_facts("Q243", max_facts=15)
        assert len(facts) == 1
        assert facts[0].value == "val1"
        self.mock_cache.set.assert_called_once()

    def test_get_entity_facts_cache_hit(self) -> None:
        cached = [
            {
                "entity": "Q243",
                "property_label": "P31",
                "value": "cached_val",
                "value_type": "literal",
            }
        ]
        self.mock_cache.get_or_sentinel.return_value = cached

        facts = self.graph.get_entity_facts("Q243")
        assert len(facts) == 1
        assert facts[0].value == "cached_val"

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_non_200(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        facts = self.graph.get_entity_facts("Q243")
        assert facts == []

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_exception(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = Exception("Network error")

        facts = self.graph.get_entity_facts("Q243")
        assert facts == []

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_skip_non_value_snak(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "entities": {
                "Q243": {
                    "claims": {
                        "P31": [
                            {
                                "mainsnak": {
                                    "snaktype": "somevalue",
                                    "datavalue": None,
                                }
                            }
                        ]
                    }
                }
            }
        }
        mock_get.return_value = mock_resp

        facts = self.graph.get_entity_facts("Q243")
        assert facts == []

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_dict_value(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "entities": {
                "Q243": {
                    "claims": {
                        "P31": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {
                                        "type": "wikibase-entityid",
                                        "value": {"id": "Q1", "type": "item"},
                                    },
                                }
                            }
                        ]
                    }
                }
            }
        }
        mock_get.return_value = mock_resp

        facts = self.graph.get_entity_facts("Q243")
        assert len(facts) == 1
        assert facts[0].value == "Q1"
        assert facts[0].value_type == "wikibase-entityid"

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_dedup_and_max_facts(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "entities": {
                "Q243": {
                    "claims": {
                        "P31": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {
                                        "type": "string",
                                        "value": "same_val",
                                    },
                                }
                            }
                        ],
                        "P32": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {
                                        "type": "string",
                                        "value": "other_val",
                                    },
                                }
                            }
                        ],
                    }
                }
            }
        }
        mock_get.return_value = mock_resp

        facts = self.graph.get_entity_facts("Q243", max_facts=1)
        assert len(facts) == 1

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_empty_claims(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"entities": {"Q243": {"claims": {}}}}
        mock_get.return_value = mock_resp

        facts = self.graph.get_entity_facts("Q243")
        assert facts == []

    def test_clean_entity_name(self) -> None:
        assert (
            WikidataGraph._clean_entity_name("  The Eiffel Tower  ") == "Eiffel Tower"
        )
        assert WikidataGraph._clean_entity_name("A Apple") == "Apple"
        assert WikidataGraph._clean_entity_name("An Orange") == "Orange"
        assert WikidataGraph._clean_entity_name("Foo") == "Foo"
        assert WikidataGraph._clean_entity_name("") == ""

    def test_get_related_entities(self) -> None:
        assert self.graph.get_related_entities("Q243") == []

    def test_enrich_with_qid(self) -> None:
        with (
            patch.object(
                self.graph, "find_entity_id", return_value="Q243"
            ) as mock_find,
            patch.object(
                self.graph,
                "get_entity_facts",
                return_value=[
                    EntityFact(
                        entity="Q243",
                        property_label="architect",
                        value="Gustave Eiffel",
                    )
                ],
            ) as mock_facts,
            patch.object(
                self.graph, "get_related_entities", return_value=["Q1"]
            ) as mock_related,
        ):
            result = self.graph.enrich("Eiffel Tower")
            assert result.entity_name == "Eiffel Tower"
            assert result.wikidata_id == "Q243"
            assert len(result.facts) == 1
            mock_find.assert_called_once_with("Eiffel Tower")

    def test_enrich_cleaned_name_with_article(self) -> None:
        with patch.object(self.graph, "find_entity_id", return_value="Q243"):
            result = self.graph.enrich("The Eiffel Tower")
            assert result.entity_name == "The Eiffel Tower"
            assert result.wikidata_id == "Q243"

    def test_enrich_no_qid(self) -> None:
        with patch.object(self.graph, "find_entity_id", return_value=None) as mock_find:
            result = self.graph.enrich("Nowhere")
            assert result.entity_name == "Nowhere"
            assert result.wikidata_id is None
            assert result.facts == []

    def test_enrich_no_qid_cleaned_empty(self) -> None:
        with patch.object(self.graph, "find_entity_id", return_value=None):
            result = self.graph.enrich("The")
            assert result.entity_name == "The"
            assert result.wikidata_id is None

    def test_enrich_all(self) -> None:
        with (
            patch.object(
                self.graph,
                "enrich",
                side_effect=lambda n: GraphResult(
                    entity_name=n, wikidata_id=f"Q{n[:2]}"
                ),
            ),
        ):
            results = self.graph.enrich_all(["Paris", "London"])
            assert len(results) == 2
            assert results[0].entity_name == "Paris"

    def test_run_query_success(self) -> None:
        self.mock_sparql.setQuery.reset_mock()
        self.mock_sparql.query.return_value.convert.return_value = {
            "results": {"bindings": [{"item": {"value": "Q1"}}]}
        }

        rows = self.graph._run_query("SELECT *")
        assert len(rows) == 1

    def test_run_query_429_retry_then_success(self) -> None:
        """A single 429 followed by a success should return results."""
        from urllib.error import HTTPError as _HTTPError

        http_error_429 = _HTTPError("url", 429, "Too Many", {}, None)
        success_result = MagicMock()
        success_result.convert.return_value = {
            "results": {"bindings": [{"item": {"value": "Q1"}}]}
        }
        self.mock_sparql.query.side_effect = [http_error_429, success_result]

        rows = self.graph._run_query("SELECT *", retries=3)
        assert len(rows) == 1
        assert rows[0]["item"]["value"] == "Q1"

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_find_entity_id_api_cache_hit_on_no_search(
        self, mock_get: MagicMock
    ) -> None:
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        qid = self.graph.find_entity_id("Eiffel Tower")
        assert qid is None

    def test_run_query_retry_then_exception(self) -> None:
        self.mock_sparql.setQuery.reset_mock()
        http_error_429 = __import__("urllib.error", fromlist=["HTTPError"]).HTTPError(
            "url", 429, "Too Many", {}, None
        )

        # Simulate 3 429s to exhaust retries
        self.mock_sparql.query.side_effect = [
            http_error_429,
            http_error_429,
            http_error_429,
        ]

        rows = self.graph._run_query("SELECT *", retries=3)
        assert rows == []

    def test_run_query_exception(self) -> None:
        self.mock_sparql.setQuery.reset_mock()
        self.mock_sparql.query.side_effect = Exception("SPARQL error")

        rows = self.graph._run_query("SELECT *")
        assert rows == []
