import pytest
from unittest.mock import MagicMock, patch

from watson_lite.core.models import Passage
from watson_lite.retrieval.bm25_retriever import (
    BM25Retriever,
    fetch_wikipedia_passages,
)


class TestFetchWikipediaPassages:
    def setup_method(self) -> None:
        self.cache_patcher = patch(
            "watson_lite.retrieval.bm25_retriever.get_cache"
        )
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        self.mock_get_cache.return_value = self.mock_cache

    def teardown_method(self) -> None:
        self.cache_patcher.stop()

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_cache_hit(self, mock_get: MagicMock) -> None:
        cached = [
            {
                "text": "cached text",
                "source": "Test",
                "url": "https://en.wikipedia.org/wiki/Test",
                "score": 0.0,
                "rank": 0,
            }
        ]
        self.mock_cache.get.return_value = cached

        result = fetch_wikipedia_passages("test")
        assert len(result) == 1
        assert result[0].text == "cached text"

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_search_api_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = Exception("Search error")

        result = fetch_wikipedia_passages("test")
        assert result == []

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_search_success_extract_error(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "query": {
                "search": [{"title": "Test Page"}]
            }
        }

        def side_effect(url, **kwargs):
            if "list=search" in str(kwargs.get("params", {})):
                return search_resp
            raise Exception("Extract error")

        mock_get.side_effect = side_effect

        # The extract params won't match -- easier to use side_effect per call
        mock_get.side_effect = None
        mock_get.return_value = search_resp

        # Override: second call raises
        extract_resp = MagicMock()
        extract_resp.json.side_effect = Exception("Parse error")
        # We need to return different values for consecutive calls
        mock_get.side_effect = [search_resp, extract_resp]

        result = fetch_wikipedia_passages("test")
        assert result == []

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_full_success(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "query": {
                "search": [{"title": "Test Article"}]
            }
        }
        extract_resp = MagicMock()
        extract_resp.json.return_value = {
            "query": {
                "pages": {
                    "1": {
                        "extract": (
                            "word " * 300
                        )
                    }
                }
            }
        }
        mock_get.side_effect = [search_resp, extract_resp]

        result = fetch_wikipedia_passages("test")
        assert len(result) > 0
        assert result[0].source == "Test Article"

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_extract_skips_empty_text(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "query": {
                "search": [{"title": "Empty"}]
            }
        }
        extract_resp = MagicMock()
        extract_resp.json.return_value = {
            "query": {
                "pages": {
                    "1": {
                        "extract": ""
                    }
                }
            }
        }
        mock_get.side_effect = [search_resp, extract_resp]

        result = fetch_wikipedia_passages("test")
        assert result == []

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_extract_skips_short_chunk(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "query": {
                "search": [{"title": "Short"}]
            }
        }
        extract_resp = MagicMock()
        extract_resp.json.return_value = {
            "query": {
                "pages": {
                    "1": {
                        "extract": "hello world"
                    }
                }
            }
        }
        mock_get.side_effect = [search_resp, extract_resp]

        result = fetch_wikipedia_passages("test")
        assert result == []


class TestBM25Retriever:
    def setup_method(self) -> None:
        self.cache_patcher = patch(
            "watson_lite.retrieval.bm25_retriever.get_cache"
        )
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        self.mock_get_cache.return_value = self.mock_cache

        self.bm25s_patcher = patch(
            "watson_lite.retrieval.bm25_retriever.bm25s"
        )
        self.mock_bm25s = self.bm25s_patcher.start()
        self.mock_bm25s.tokenize.return_value = "tokenized_corpus"
        self.mock_retriever_instance = MagicMock()
        self.mock_bm25_cls = MagicMock(return_value=self.mock_retriever_instance)
        self.mock_bm25s.BM25 = self.mock_bm25_cls

        self.retriever = BM25Retriever()

    def teardown_method(self) -> None:
        self.cache_patcher.stop()
        self.bm25s_patcher.stop()

    def test_index(self) -> None:
        passages = [
            Passage(
                text="Paris is the capital of France.",
                source="Paris",
                url="http://example.com",
            ),
            Passage(
                text="London is the capital of the UK.",
                source="London",
                url="http://example.com",
            ),
        ]
        self.retriever.index(passages)
        assert len(self.retriever.passages) == 2
        self.mock_bm25s.tokenize.assert_called_once()
        self.mock_bm25_cls.assert_called_once()
        self.mock_retriever_instance.index.assert_called_once_with(
            "tokenized_corpus"
        )

    def test_retrieve_empty(self) -> None:
        result = self.retriever. retrieve("test")
        assert result == []

    def test_retrieve_with_results(self) -> None:
        passages = [
            Passage(
                text="unique passage text",
                source="Src",
                url="http://example.com",
            ),
        ]
        self.retriever.passages = passages
        self.mock_retriever = MagicMock()
        self.retriever.retriever = self.mock_retriever

        self.mock_bm25s.tokenize.return_value = "tokenized_query"
        self.mock_retriever.retrieve.return_value = (
            [["unique passage text"]],
            [[0.95]],
        )

        result = self.retriever.retrieve("test query", top_k=10)
        assert len(result) == 1
        assert result[0].text == "unique passage text"
        assert result[0].score == 0.95
        assert result[0].rank == 1

    def test_retrieve_no_match_in_passage_map(self) -> None:
        passages = [
            Passage(
                text="doc text",
                source="Src",
                url="http://example.com",
            ),
        ]
        self.retriever.passages = passages
        self.mock_retriever = MagicMock()
        self.retriever.retriever = self.mock_retriever

        self.mock_bm25s.tokenize.return_value = "tokenized_query"
        self.mock_retriever.retrieve.return_value = (
            [["non-existent text"]],
            [[0.5]],
        )

        result = self.retriever.retrieve("query")
        assert result == []

    def test_fetch_and_retrieve_empty_passages(self) -> None:
        with patch(
            "watson_lite.retrieval.bm25_retriever.fetch_wikipedia_passages",
            return_value=[],
        ):
            result = self.retriever.fetch_and_retrieve("test")
            assert result == []

    def test_fetch_and_retrieve_full_flow(self) -> None:
        passages = [
            Passage(
                text="Paris is great.",
                source="Paris",
                url="http://example.com",
            ),
        ]
        with patch(
            "watson_lite.retrieval.bm25_retriever.fetch_wikipedia_passages",
            return_value=passages,
        ):
            self.mock_bm25s.tokenize.return_value = "tok"
            self.mock_bm25_cls.return_value = self.mock_retriever_instance
            self.mock_retriever_instance.retrieve.return_value = (
                [["Paris is great."]],
                [[0.9]],
            )

            result = self.retriever.fetch_and_retrieve("Paris", top_k=5)
            assert len(result) == 1
            assert result[0].text == "Paris is great."
