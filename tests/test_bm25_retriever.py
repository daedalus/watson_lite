from unittest.mock import MagicMock, patch

import pytest

from watson_lite.core.cache import SENTINEL
from watson_lite.core.models import Passage
from watson_lite.retrieval.bm25_retriever import (
    BM25Retriever,
    fetch_arxiv_passages,
    fetch_dbpedia_passages,
    fetch_elasticsearch_passages,
    fetch_huggingface_passages,
    fetch_oeis_passages,
    fetch_openlibrary_passages,
    fetch_pubmed_passages,
    fetch_stackexchange_passages,
    fetch_wikinews_passages,
    fetch_wikiquote_passages,
    fetch_wikisource_passages,
    fetch_wikipedia_passages,
)


class TestFetchWikipediaPassages:
    def setup_method(self) -> None:
        self.cache_patcher = patch("watson_lite.retrieval.bm25_retriever.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        # Default: every key is a cache miss.
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache
        self.sleep_patcher = patch("watson_lite.retrieval.bm25_retriever.time.sleep")
        self.sleep_patcher.start()

    def teardown_method(self) -> None:
        self.cache_patcher.stop()
        self.sleep_patcher.stop()

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
        self.mock_cache.get_or_sentinel.return_value = cached

        result = fetch_wikipedia_passages("test")
        assert len(result) == 1
        assert result[0].text == "cached text"

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_search_api_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = Exception("Search error")

        result = fetch_wikipedia_passages("test")
        assert result == []
        self.mock_cache.set.assert_called_once_with(
            "wiki:passages:test:top_k=5",
            [],
            ttl_seconds=300,
        )

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_search_success_extract_error(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.json.return_value = {"query": {"search": [{"title": "Test Page"}]}}

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
            "query": {"search": [{"title": "Test Article"}]}
        }
        extract_resp = MagicMock()
        extract_resp.json.return_value = {
            "query": {"pages": {"1": {"extract": ("word " * 300)}}}
        }
        mock_get.side_effect = [search_resp, extract_resp]

        result = fetch_wikipedia_passages("test")
        assert len(result) > 0
        assert result[0].source == "Test Article"

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_extract_skips_empty_text(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.json.return_value = {"query": {"search": [{"title": "Empty"}]}}
        extract_resp = MagicMock()
        extract_resp.json.return_value = {"query": {"pages": {"1": {"extract": ""}}}}
        mock_get.side_effect = [search_resp, extract_resp]

        result = fetch_wikipedia_passages("test")
        assert result == []

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_extract_skips_short_chunk(self, mock_get: MagicMock) -> None:
        search_resp = MagicMock()
        search_resp.json.return_value = {"query": {"search": [{"title": "Short"}]}}
        extract_resp = MagicMock()
        extract_resp.json.return_value = {
            "query": {"pages": {"1": {"extract": "hello world"}}}
        }
        mock_get.side_effect = [search_resp, extract_resp]

        result = fetch_wikipedia_passages("test")
        assert result == []


class TestBM25Retriever:
    def setup_method(self) -> None:
        self.cache_patcher = patch("watson_lite.retrieval.bm25_retriever.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache

        self.bm25s_patcher = patch("watson_lite.retrieval.bm25_retriever.bm25s")
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
        self.mock_retriever_instance.index.assert_called_once_with("tokenized_corpus")

    def test_retrieve_empty(self) -> None:
        result = self.retriever.retrieve("test")
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


class TestFetchElasticsearchPassages:
    def setup_method(self) -> None:
        self.cache_patcher = patch("watson_lite.retrieval.bm25_retriever.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache

    def teardown_method(self) -> None:
        self.cache_patcher.stop()

    @patch("watson_lite.retrieval.bm25_retriever.requests.post")
    def test_missing_configuration_returns_empty(self, mock_post: MagicMock) -> None:
        result = fetch_elasticsearch_passages(
            "python",
            base_url="",
            index="",
        )
        assert result == []
        mock_post.assert_not_called()

    @patch("watson_lite.retrieval.bm25_retriever.requests.post")
    def test_cache_hit(self, mock_post: MagicMock) -> None:
        cached = [
            {
                "text": "cached text",
                "source": "Doc",
                "url": "https://example.org/doc",
                "score": 0.0,
                "rank": 0,
            }
        ]
        self.mock_cache.get_or_sentinel.return_value = cached
        result = fetch_elasticsearch_passages(
            "python",
            base_url="http://localhost:9200",
            index="passages",
        )
        assert len(result) == 1
        assert result[0].text == "cached text"
        mock_post.assert_not_called()

    @patch("watson_lite.retrieval.bm25_retriever.requests.post")
    def test_success(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_source": {
                            "title": "Python",
                            "text": "Python is a programming language.",
                            "url": "https://example.org/python",
                        },
                    }
                ]
            }
        }
        mock_post.return_value = response

        result = fetch_elasticsearch_passages(
            "python",
            base_url="http://localhost:9200",
            index="passages",
        )

        assert len(result) == 1
        assert result[0].source == "Python"
        assert result[0].text == "Python is a programming language."


class TestFetchHuggingFacePassages:
    def setup_method(self) -> None:
        self.cache_patcher = patch("watson_lite.retrieval.bm25_retriever.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache

    def teardown_method(self) -> None:
        self.cache_patcher.stop()

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_missing_configuration_returns_empty(self, mock_get: MagicMock) -> None:
        result = fetch_huggingface_passages(
            "python",
            dataset="",
            split="",
        )
        assert result == []
        mock_get.assert_not_called()

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_cache_hit(self, mock_get: MagicMock) -> None:
        cached = [
            {
                "text": "cached text",
                "source": "ag_news",
                "url": "https://huggingface.co/datasets/ag_news",
                "score": 0.0,
                "rank": 0,
            }
        ]
        self.mock_cache.get_or_sentinel.return_value = cached
        result = fetch_huggingface_passages(
            "python",
            dataset="ag_news",
            split="train",
        )
        assert len(result) == 1
        assert result[0].text == "cached text"
        mock_get.assert_not_called()

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "rows": [
                {
                    "row": {
                        "title": "Sample row",
                        "text": "Python is a programming language used for many tasks.",
                    }
                }
            ]
        }
        mock_get.return_value = response

        result = fetch_huggingface_passages(
            "python",
            dataset="ag_news",
            config="default",
            split="train",
        )

        assert len(result) == 1
        assert result[0].source == "Sample row"
        assert result[0].text == "Python is a programming language used for many tasks."


class TestAdditionalPublicSources:
    def setup_method(self) -> None:
        self.cache_patcher = patch("watson_lite.retrieval.bm25_retriever.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache

    def teardown_method(self) -> None:
        self.cache_patcher.stop()

    @patch("watson_lite.retrieval.bm25_retriever.fetch_mediawiki_passages")
    def test_wikiquote_delegates_to_mediawiki(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = []
        fetch_wikiquote_passages("python", top_k=3)
        mock_fetch.assert_called_once()
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs["api_url"] == "https://en.wikiquote.org/w/api.php"
        assert kwargs["cache_namespace"] == "wikiquote"

    @patch("watson_lite.retrieval.bm25_retriever.fetch_mediawiki_passages")
    def test_wikisource_delegates_to_mediawiki(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = []
        fetch_wikisource_passages("python", top_k=3)
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs["api_url"] == "https://en.wikisource.org/w/api.php"
        assert kwargs["cache_namespace"] == "wikisource"

    @patch("watson_lite.retrieval.bm25_retriever.fetch_mediawiki_passages")
    def test_wikinews_delegates_to_mediawiki(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = []
        fetch_wikinews_passages("python", top_k=3)
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs["api_url"] == "https://en.wikinews.org/w/api.php"
        assert kwargs["cache_namespace"] == "wikinews"

    @patch("watson_lite.retrieval.bm25_retriever._request_json")
    def test_pubmed_success(self, mock_request_json: MagicMock) -> None:
        mock_request_json.side_effect = [
            {"esearchresult": {"idlist": ["12345"]}},
            {
                "result": {
                    "uids": ["12345"],
                    "12345": {
                        "title": "Python in medicine",
                        "fulljournalname": "Medical Journal",
                        "pubdate": "2025",
                        "authors": [{"name": "Jane Doe"}],
                    },
                }
            },
        ]

        result = fetch_pubmed_passages("python")

        assert len(result) == 1
        assert result[0].source == "Python in medicine"
        assert result[0].url == "https://pubmed.ncbi.nlm.nih.gov/12345/"

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_arxiv_success(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678</id>
    <title>Python Research</title>
    <summary>Python is used for scientific computing and reproducible research workflows.</summary>
  </entry>
</feed>
"""
        mock_get.return_value = response

        result = fetch_arxiv_passages("python")

        assert len(result) == 1
        assert result[0].source == "Python Research"

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_openlibrary_success(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "docs": [
                {
                    "title": "Learning Python",
                    "key": "/works/OL1W",
                    "author_name": ["Mark Lutz"],
                    "first_sentence": ["A guide to Python."],
                    "subject": ["programming"],
                    "first_publish_year": 1999,
                }
            ]
        }
        mock_get.return_value = response

        result = fetch_openlibrary_passages("python")

        assert len(result) == 1
        assert result[0].source == "Learning Python"
        assert result[0].url == "https://openlibrary.org/works/OL1W"

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_stackexchange_success(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "items": [
                {
                    "title": "How to use Python lists?",
                    "body": "<p>Use append and extend methods.</p>",
                    "tags": ["python", "list"],
                    "link": "https://stackoverflow.com/questions/1",
                }
            ]
        }
        mock_get.return_value = response

        result = fetch_stackexchange_passages("python")

        assert len(result) == 1
        assert result[0].source == "StackExchange:stackoverflow"
        assert result[0].url == "https://stackoverflow.com/questions/1"

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_dbpedia_success(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "docs": [
                {
                    "label": ["Python_(programming_language)"],
                    "comment": ["Python is an interpreted language."],
                    "resource": [
                        "http://dbpedia.org/resource/Python_(programming_language)"
                    ],
                    "typeName": ["ProgrammingLanguage"],
                }
            ]
        }
        mock_get.return_value = response

        result = fetch_dbpedia_passages("python")

        assert len(result) == 1
        assert "Python_(programming_language)" in result[0].source

    @patch("watson_lite.retrieval.bm25_retriever.requests.get")
    def test_oeis_success(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "results": [
                {
                    "number": 45,
                    "name": "Fibonacci numbers",
                    "data": "1,1,2,3,5,8",
                    "comment": "Classic sequence",
                }
            ]
        }
        mock_get.return_value = response

        result = fetch_oeis_passages("fibonacci")

        assert len(result) == 1
        assert result[0].source == "A000045"
        assert result[0].url == "https://oeis.org/A000045"
