"""Adversarial and edge-case tests — inputs the pipeline should survive."""

import json
import math
import os
import re
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from watson_lite.core.cache import SENTINEL, Cache, is_cache_miss
from watson_lite.core.extractor import ConfidenceScorer, _question_type_bonus
from watson_lite.core.models import (
    AnswerCandidate,
    EntityFact,
    FinalAnswer,
    GraphResult,
    Passage,
    RankedPassage,
)
from watson_lite.graph.wikidata import WikidataGraph
from watson_lite.ranking.ranker import Ranker, RRFFusion
from watson_lite.retrieval.bm25_retriever import BM25Retriever
from watson_lite.retrieval.vector_retriever import VectorRetriever

# ---------------------------------------------------------------------------
# Cache — serialisation, corruption, legacy format
# ---------------------------------------------------------------------------


class TestCacheAdversarial:
    def setup_method(self) -> None:
        fd, self.tmp = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        self.cache = Cache(self.tmp)

    def teardown_method(self) -> None:
        self.cache.close()
        os.unlink(self.tmp)

    def _inject_and_reopen(self, key: str, raw_value: str) -> None:
        """Inject raw data into SQLite bypassing Cache, then reopen so bloom picks it up."""
        self.cache.close()
        con = sqlite3.connect(self.tmp)
        con.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, raw_value, 0.0),
        )
        con.commit()
        con.close()
        self.cache = Cache(self.tmp)

    def test_unwrap_legacy_bare_string(self) -> None:
        self.cache.set("k", "raw_string")
        row = self.cache.con.execute(
            "SELECT value FROM cache WHERE key = 'k'"
        ).fetchone()
        raw = row[0]
        wrapped = json.loads(raw)
        assert isinstance(wrapped, dict) and "v" in wrapped
        # Manually overwrite with legacy format (bare value, not wrapped).
        self.cache.con.execute(
            "UPDATE cache SET value = ? WHERE key = 'k'", (json.dumps("legacy"),)
        )
        self.cache.con.commit()
        assert self.cache.get("k") == "legacy"

    def test_unwrap_legacy_bare_int(self) -> None:
        self._inject_and_reopen("int_key", json.dumps(42))
        assert self.cache.get("int_key") == 42
        assert self.cache.get_or_sentinel("int_key") == 42

    def test_unwrap_legacy_bare_list(self) -> None:
        self._inject_and_reopen("list_key", json.dumps(["a", "b"]))
        assert self.cache.get("list_key") == ["a", "b"]

    def test_unwrap_legacy_bare_null(self) -> None:
        self._inject_and_reopen("null_key", json.dumps(None))
        assert self.cache.get("null_key") is None
        assert self.cache.get_or_sentinel("null_key") is None

    def test_non_serializable_value_falls_back_to_str(self) -> None:
        """set() passes default=str to json.dumps so non-serialisable types become str."""
        self.cache.set("set_key", {1, 2, 3})
        val = self.cache.get("set_key")
        assert isinstance(val, str)

    def test_corrupted_json_raises(self) -> None:
        self._inject_and_reopen("bad", "not valid json[[[")
        with pytest.raises(json.JSONDecodeError):
            self.cache.get("bad")

    def test_clear_on_empty_cache_succeeds(self) -> None:
        c = Cache(tempfile.mktemp(suffix=".sqlite3"))
        c.clear()
        c.close()
        os.unlink(c.db_path)

    def test_empty_key(self) -> None:
        self.cache.set("", "empty_key")
        assert self.cache.get("") == "empty_key"

    def test_very_large_value(self) -> None:
        large = "x" * 100_000
        self.cache.set("large", large)
        assert self.cache.get("large") == large

    def test_sentinel_is_singleton(self) -> None:
        assert SENTINEL is SENTINEL
        assert is_cache_miss(SENTINEL) is True
        assert is_cache_miss(None) is False
        assert is_cache_miss("miss") is False
        assert is_cache_miss(0) is False


# ---------------------------------------------------------------------------
# NLP — broken / extreme / adversarial inputs
# ---------------------------------------------------------------------------


class TestNLPAdversarial:
    def test_classify_question_empty(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        assert nlp.classify_question("") == "unknown"
        assert nlp.classify_question("   ") == "unknown"

    def test_classify_question_gibberish(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        assert nlp.classify_question("xyzzx") == "unknown"

    def test_classify_question_unicode(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        assert nlp.classify_question("¿Quién diseñó la Torre Eiffel?") == "unknown"
        assert nlp.classify_question("谁建造了埃菲尔铁塔？") == "unknown"

    def test_extract_keywords_empty_input(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        doc = nlp.nlp("")
        assert nlp.extract_keywords(doc) == []

    def test_extract_keywords_short_tokens_excluded(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        doc = nlp.nlp("I am a cat")
        for kw in nlp.extract_keywords(doc):
            assert len(kw) > 2

    def test_decompose_question_no_conjunction(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        assert nlp.decompose_question("Why?") == ["Why?"]

    def test_decompose_question_trailing_conjunction(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        result = nlp.decompose_question("Who did it and")
        assert len(result) == 1

    def test_process_very_long_question(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        long_q = "What " + "is " * 500 + "? "
        parsed = nlp.process(long_q)
        assert parsed.question_type == "what"

    def test_process_empty_question_does_not_crash(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        parsed = nlp.process("")
        assert parsed.raw == ""


# ---------------------------------------------------------------------------
# BM25 — broken / adversarial inputs
# ---------------------------------------------------------------------------


class TestBM25Adversarial:
    def setup_method(self) -> None:
        self.cache_patcher = patch("watson_lite.retrieval.bm25_retriever.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        self.mock_get_cache.return_value = self.mock_cache

        self.bm25s_patcher = patch("watson_lite.retrieval.bm25_retriever.bm25s")
        self.mock_bm25s = self.bm25s_patcher.start()
        self.mock_bm25s.tokenize.return_value = "tokenized_corpus"
        self.mock_retriever_instance = MagicMock()
        self.mock_bm25s.BM25.return_value = self.mock_retriever_instance

        self.retriever = BM25Retriever()

    def teardown_method(self) -> None:
        self.cache_patcher.stop()
        self.bm25s_patcher.stop()

    def test_retrieve_empty_query(self) -> None:
        self.retriever.index([Passage("hello world", "s", "u")])
        self.mock_retriever_instance.retrieve.return_value = ([], [])
        with pytest.raises(IndexError):
            self.retriever.retrieve("", top_k=10)

    def test_index_empty_corpus(self) -> None:
        self.retriever.index([])
        self.mock_retriever_instance.retrieve.return_value = ([], [])
        result = self.retriever.retrieve("test", top_k=10)
        assert result == []


# ---------------------------------------------------------------------------
# Vector retriever — adversarial inputs
# ---------------------------------------------------------------------------


class TestVectorAdversarial:
    def setup_method(self) -> None:
        self.faiss_patcher = patch("watson_lite.retrieval.vector_retriever.faiss")
        self.mock_faiss = self.faiss_patcher.start()
        self.mock_index = MagicMock()
        self.mock_index.ntotal = 2
        self.mock_faiss.IndexFlatIP.return_value = self.mock_index

        self.model_patcher = patch(
            "watson_lite.retrieval.vector_retriever.SentenceTransformer"
        )
        self.mock_model_cls = self.model_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model_cls.return_value = self.mock_model
        self.mock_model.get_sentence_embedding_dimension.return_value = 384

        self.retriever = VectorRetriever()

    def teardown_method(self) -> None:
        self.faiss_patcher.stop()
        self.model_patcher.stop()

    def test_index_empty_corpus(self) -> None:
        self.mock_model.encode.return_value = np.empty((0, 384), dtype="float32")
        self.retriever.index_passages([])
        assert len(self.retriever.passages) == 0

    def test_retrieve_top_k_zero(self) -> None:
        passage = Passage("hello world", "s", "u")
        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")
        self.retriever.index_passages([passage])
        with pytest.raises(ValueError):
            self.retriever.retrieve("test", top_k=0)

    def test_query_embedding_dimension_mismatch(self) -> None:
        passage = Passage("hello world", "s", "u")
        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")
        self.retriever.index_passages([passage])

        self.mock_model.encode.return_value = np.array([[0.1] * 128], dtype="float32")
        with pytest.raises(Exception):
            self.retriever.retrieve("test", top_k=10)

    def test_faiss_returns_negative_indices(self) -> None:
        passage = Passage("hello world", "s", "u")
        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")
        self.retriever.index_passages([passage])

        self.mock_model.encode.side_effect = None
        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")
        self.mock_index.search.return_value = (
            np.array([[0.5]]),
            np.array([[-1]]),
        )
        result = self.retriever.retrieve("test", top_k=10)
        assert result == []


# ---------------------------------------------------------------------------
# Wikidata graph — adversarial / unusual inputs
# ---------------------------------------------------------------------------


class TestWikidataAdversarial:
    def setup_method(self) -> None:
        self.sparql_patcher = patch("watson_lite.graph.wikidata.SPARQLWrapper")
        self.mock_sparql_cls = self.sparql_patcher.start()
        self.mock_sparql = MagicMock()
        self.mock_sparql_cls.return_value = self.mock_sparql

        self.cache_patcher = patch("watson_lite.graph.wikidata.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache

        self.graph = WikidataGraph()

    def teardown_method(self) -> None:
        self.sparql_patcher.stop()
        self.cache_patcher.stop()

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_find_entity_id_sparql_injection_attempts(
        self, mock_get: MagicMock
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        self.mock_sparql.query().convert.return_value = {"results": {"bindings": []}}

        injections = [
            '"; DROP TABLE cache; --',
            ' " OR 1=1 --',
            "\\",
            '"; SELECT * FROM information_schema.tables; --',
            "a" * 1000,
            "\x00",
            "\n' OR '1'='1",
        ]
        for payload in injections:
            result = self.graph.find_entity_id(payload)
            assert result is None or isinstance(result, str)

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_missing_keys(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"entities": {"Q999": {}}}
        mock_get.return_value = mock_resp

        facts = self.graph.get_entity_facts("Q999")
        assert facts == []

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_get_entity_facts_partial_resolve_failure(
        self, mock_get: MagicMock
    ) -> None:
        entity_resp = MagicMock()
        entity_resp.status_code = 200
        entity_resp.json.return_value = {
            "entities": {
                "Q243": {
                    "claims": {
                        "P84": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {
                                        "value": {"id": "Q207773"},
                                        "type": "wikibase-entityid",
                                    },
                                }
                            }
                        ],
                        "P131": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {
                                        "value": {"id": "Q99999999"},
                                        "type": "wikibase-entityid",
                                    },
                                }
                            }
                        ],
                    }
                }
            }
        }

        label_resp = MagicMock()
        label_resp.status_code = 200
        label_resp.json.return_value = {
            "entities": {
                "Q207773": {"labels": {"en": {"value": "Gustave Eiffel"}}},
            }
        }

        mock_get.side_effect = [entity_resp, label_resp]
        facts = self.graph.get_entity_facts("Q243")
        labels = [f.value for f in facts]
        assert "Gustave Eiffel" in labels
        assert "Q99999999" in labels

    def test_clean_entity_name_edge_cases(self) -> None:
        assert WikidataGraph._clean_entity_name("the") == "the"
        assert WikidataGraph._clean_entity_name("a") == "a"
        assert WikidataGraph._clean_entity_name("an") == "an"
        assert WikidataGraph._clean_entity_name("") == ""
        assert WikidataGraph._clean_entity_name("  ") == ""
        assert WikidataGraph._clean_entity_name("The The") == "The"
        assert WikidataGraph._clean_entity_name("A A") == "A"

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_enrich_all_empty(self, mock_get: MagicMock) -> None:
        result = self.graph.enrich_all([])
        assert result == []

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_enrich_entity_with_non_string_value_in_fact(
        self, mock_get: MagicMock
    ) -> None:
        self.mock_cache.get_or_sentinel.return_value = [
            {
                "entity": "Q1",
                "property_label": "coordinate",
                "value": 123.45,
                "value_type": "number",
            },
            {
                "entity": "Q1",
                "property_label": "inception",
                "value": None,
                "value_type": "time",
            },
        ]
        related = self.graph.get_related_entities("Q1")
        assert related == []


# ---------------------------------------------------------------------------
# Ranker — edge cases in fusion and cross-encoder scoring
# ---------------------------------------------------------------------------


class TestRankerAdversarial:
    def test_rrf_fuse_empty_lists(self) -> None:
        result = RRFFusion().fuse([], k=60)
        assert result == []

    def test_rrf_fuse_single_list(self) -> None:
        ps = [Passage(f"text{i}", "s", "u", rank=i) for i in range(3)]
        result = RRFFusion().fuse([ps], k=60)
        assert len(result) == 3

    def test_rrf_fuse_top_k_zero(self) -> None:
        bm25 = [Passage("a", "s", "u", rank=1)]
        vec = [Passage("a", "s", "u", rank=1)]
        result = RRFFusion().fuse([bm25, vec], k=60)
        result = result[:0]
        assert result == []

    def test_cross_encoder_nan_score(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([math.nan])

        from watson_lite.ranking.ranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker.model = mock_model
        ps = [Passage("hello world", "s", "u", rank=1)]
        result = reranker.rerank("q", ps, top_k=10)
        assert len(result) == 1
        assert math.isnan(result[0].cross_score)


# ---------------------------------------------------------------------------
# Confidence scorer — edge-case inputs
# ---------------------------------------------------------------------------


class TestConfidenceScorerAdversarial:
    def test_empty_candidates(self) -> None:
        scorer = ConfidenceScorer()
        answer = scorer.score([], [], "who")
        assert answer.answer == "No answer found"
        assert answer.confidence == 0.0
        assert "reason" in answer.confidence_breakdown

    def test_candidates_with_zero_scores(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="x", source="s", url="", passage="", extraction_score=0.0, rank=1
            )
        ]
        answer = scorer.score(candidates, [], "what")
        assert answer.confidence == 0.25

    def test_all_scores_zero(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="x", source="s", url="", passage="", extraction_score=0.0, rank=100
            ),
            AnswerCandidate(
                span="y", source="s", url="", passage="", extraction_score=0.0, rank=200
            ),
        ]
        answer = scorer.score(candidates, [], "what")
        assert answer.confidence == 0.1

    def test_span_agreement_all_unique(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span=f"ans{i}",
                source="s",
                url="",
                passage="",
                extraction_score=float(1 - i * 0.1),
                rank=i + 1,
            )
            for i in range(5)
        ]
        answer = scorer.score(candidates, [], "what")
        assert answer.confidence_breakdown["span_agreement"] == pytest.approx(0.2)

    def test_rank_penalty_beyond_10(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="x", source="s", url="", passage="", extraction_score=0.9, rank=15
            )
        ]
        answer = scorer.score(candidates, [], "what")
        assert answer.confidence_breakdown["passage_rank_signal"] == 0.0

    def test_graph_corroboration_case_insensitive(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            )
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="architect",
                        value="gustave eiffel",
                    )
                ],
            )
        ]
        answer = scorer.score(candidates, graph_results, "who")
        assert answer.confidence_breakdown["graph_corroboration"] == 0.2

    def test_question_type_bonus_when_no_match(self) -> None:
        assert _question_type_bonus("foo", "how") == 0.0
        assert _question_type_bonus("bar", "why") == 0.0
        assert _question_type_bonus("", "who") == 0.0

    def test_question_type_bonus_who_single_word(self) -> None:
        assert _question_type_bonus("Paris", "who") == 0.0

    def test_confidence_clamped_at_1(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="s",
                url="",
                passage="",
                extraction_score=1.0,
                rank=1,
            ),
            AnswerCandidate(
                span="Gustave Eiffel",
                source="s",
                url="",
                passage="",
                extraction_score=1.0,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="architect",
                        value="Gustave Eiffel",
                    )
                ],
            )
        ]
        answer = scorer.score(candidates, graph_results, "who")
        assert answer.confidence <= 1.0

    def test_final_answer_supporting_passages_truncation(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="x",
                source="s",
                url="",
                passage="x" * 500,
                extraction_score=0.9,
                rank=1,
            )
        ]
        answer = scorer.score(candidates, [], "what")
        assert all(len(p) <= 200 for p in answer.supporting_passages)

    def test_graph_facts_limited_to_5(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            )
        ]
        graph_results = [
            GraphResult(
                entity_name="X",
                wikidata_id="Q1",
                facts=[
                    EntityFact(entity="Q1", property_label=f"p{i}", value="Paris")
                    for i in range(10)
                ],
            )
        ]
        answer = scorer.score(candidates, graph_results, "where")
        assert len(answer.graph_facts) <= 5


# ---------------------------------------------------------------------------
# Pipeline — error propagation and edge-case orchestration
# ---------------------------------------------------------------------------


class TestPipelineAdversarial:
    def setup_method(self) -> None:
        self.patches = [
            patch("watson_lite.pipeline.NLPProcessor"),
            patch("watson_lite.pipeline.BM25Retriever"),
            patch("watson_lite.pipeline.VectorRetriever"),
            patch("watson_lite.pipeline.WikidataGraph"),
            patch("watson_lite.pipeline.Ranker"),
            patch("watson_lite.pipeline.ExtractiveReader"),
            patch("watson_lite.pipeline.ConfidenceScorer"),
        ]
        self.mocks = [p.start() for p in self.patches]
        (
            self.mock_nlp_cls,
            self.mock_bm25_cls,
            self.mock_vector_cls,
            self.mock_graph_cls,
            self.mock_ranker_cls,
            self.mock_reader_cls,
            self.mock_scorer_cls,
        ) = self.mocks

        self.mock_nlp = MagicMock()
        self.mock_nlp_cls.return_value = self.mock_nlp
        self.mock_bm25 = MagicMock()
        self.mock_bm25_cls.return_value = self.mock_bm25
        self.mock_vector = MagicMock()
        self.mock_vector_cls.return_value = self.mock_vector
        self.mock_graph = MagicMock()
        self.mock_graph_cls.return_value = self.mock_graph
        self.mock_ranker = MagicMock()
        self.mock_ranker_cls.return_value = self.mock_ranker
        self.mock_reader = MagicMock()
        self.mock_reader_cls.return_value = self.mock_reader
        self.mock_scorer = MagicMock()
        self.mock_scorer_cls.return_value = self.mock_scorer

        self.fetch_patcher = patch(
            "watson_lite.retrieval.dataset_plugins.fetch_wikipedia_passages"
        )
        self.mock_fetch = self.fetch_patcher.start()

        from watson_lite.pipeline import WatsonLite

        self.pipeline = WatsonLite()

    def teardown_method(self) -> None:
        self.fetch_patcher.stop()
        for p in self.patches:
            p.stop()

    def test_answer_empty_question_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            self.pipeline.answer("")

    def test_answer_whitespace_only_does_not_raise(self) -> None:
        self.mock_fetch.return_value = []
        result = self.pipeline.answer("   ")
        assert result is not None

    def test_verbose_logging_does_not_crash(self) -> None:
        from watson_lite.core.models import ParsedQuestion

        self.mock_fetch.return_value = [Passage("test", "s", "u")]
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test question",
            question_type="test",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test question"],
            keywords=[],
        )
        self.mock_reader.extract.return_value = [
            AnswerCandidate(
                span="ans", source="s", url="", passage="", extraction_score=0.9, rank=1
            )
        ]
        self.mock_scorer.score.return_value = FinalAnswer(
            answer="ans", confidence=0.5, source="s", url=""
        )
        result = self.pipeline.answer("test question", verbose=True)
        assert result.answer == "ans"

    def test_no_passages_returns_early(self) -> None:
        self.mock_fetch.return_value = []
        result = self.pipeline.answer("test")
        assert "Could not retrieve" in result.answer
        assert result.confidence == 0.0

    def test_graph_enrich_returns_empty(self) -> None:
        from watson_lite.core.models import ParsedQuestion

        self.mock_fetch.return_value = [Passage("test text", "s", "u")]
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test",
            question_type="test",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test"],
            keywords=[],
        )
        self.mock_ranker.rank.return_value = []
        self.mock_reader.extract.return_value = []
        self.mock_scorer.score.return_value = FinalAnswer(
            answer="ans", confidence=0.5, source="s", url=""
        )
        result = self.pipeline.answer("test")
        assert result is not None

    def test_second_call_uses_cached_passage_hash(self) -> None:
        from watson_lite.pipeline import WatsonLite, _passage_content_key

        p = Passage("hello world", "src", "url")
        h1 = _passage_content_key(p)
        h2 = _passage_content_key(p)
        assert h1 == h2

    def test_passages_hash_differs_with_different_source(self) -> None:
        from watson_lite.pipeline import _passage_content_key

        h1 = _passage_content_key(Passage("text", "src1", "url"))
        h2 = _passage_content_key(Passage("text", "src2", "url"))
        assert h1 != h2

    def test_passages_hash_differs_with_different_url(self) -> None:
        from watson_lite.pipeline import _passage_content_key

        h1 = _passage_content_key(Passage("text", "src", "url1"))
        h2 = _passage_content_key(Passage("text", "src", "url2"))
        assert h1 != h2


# ---------------------------------------------------------------------------
# _question_type_bonus — standalone edge cases
# ---------------------------------------------------------------------------


class TestQuestionTypeBonus:
    def test_when_with_year_pattern(self) -> None:
        assert _question_type_bonus("1889", "when") == 0.1
        assert _question_type_bonus("in 1889", "when") == 0.1
        assert _question_type_bonus("999", "when") == 0.0
        assert _question_type_bonus("3000", "when") == 0.0

    def test_when_with_date_word(self) -> None:
        assert _question_type_bonus("January", "when") == 0.1
        assert _question_type_bonus("March 5", "when") == 0.1
        assert _question_type_bonus("15 August 2023", "when") == 0.1

    def test_when_no_match(self) -> None:
        assert _question_type_bonus("Paris", "when") == 0.0
        assert _question_type_bonus("", "when") == 0.0


# ---------------------------------------------------------------------------
# Cache coverage — default db path, singleton
# ---------------------------------------------------------------------------


class TestCacheCoverage:
    def test_default_db_path_creates_dir(self) -> None:
        from pathlib import Path
        import tempfile

        with (
            patch(
                "watson_lite.core.cache._DEFAULT_CACHE_DIR", Path(tempfile.mkdtemp())
            ),
        ):
            from watson_lite.core.cache import Cache

            c = Cache()
            p = Path(c.db_path)
            assert p.parent.exists()
            c.close()
            p.unlink()
            p.parent.rmdir()

    def test_get_cache_singleton(self) -> None:
        import watson_lite.core.cache as cache_mod

        saved = cache_mod._cache
        cache_mod._cache = None
        try:
            c1 = cache_mod.get_cache()
            c2 = cache_mod.get_cache()
            assert c1 is c2
            c1.close()
        finally:
            cache_mod._cache = saved


# ---------------------------------------------------------------------------
# NLP coverage — decompose_question trailing chunk > 2 words
# ---------------------------------------------------------------------------


class TestNLPCoverage:
    def test_decompose_trailing_chunk_longer_than_2(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        result = nlp.decompose_question(
            "Who built the Eiffel Tower and when was it completed"
        )
        assert len(result) == 2
        assert "when was it completed" in result

    def test_decompose_single_chunk_after_conjunction(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        result = nlp.decompose_question("Who did it and when")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# ExtractiveReader coverage — exception handler, graph_corroboration or-branch
# ---------------------------------------------------------------------------


class TestExtractorCoverage:
    def test_reader_exception_skips_passage(self) -> None:
        from watson_lite.core.extractor import ExtractiveReader

        with patch("watson_lite.core.extractor.hf_pipeline") as mock_pipeline_cls:
            mock_qa = MagicMock()
            mock_qa.side_effect = RuntimeError("QA model failed")
            mock_pipeline_cls.return_value = mock_qa

            reader = ExtractiveReader()
            reader.qa = mock_qa
            rp = RankedPassage(
                passage=Passage("test text", "s", "u"),
                rank=1,
            )
            result = reader.extract("test question", [rp], top_k=5)
            assert result == []

    def test_graph_corroboration_second_branch(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            )
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="architect",
                        value="Gustave",  # shorter value contained in span
                    )
                ],
            )
        ]
        answer = scorer.score(candidates, graph_results, "who")
        assert answer.confidence_breakdown["graph_corroboration"] == 0.2


# ---------------------------------------------------------------------------
# Wikidata coverage — SPARQL exhaustion, _resolve_qid_labels edge cases, enrich
# ---------------------------------------------------------------------------


class TestWikidataCoverage:
    def setup_method(self) -> None:
        self.sparql_patcher = patch("watson_lite.graph.wikidata.SPARQLWrapper")
        self.mock_sparql_cls = self.sparql_patcher.start()
        self.mock_sparql = MagicMock()
        self.mock_sparql_cls.return_value = self.mock_sparql

        self.cache_patcher = patch("watson_lite.graph.wikidata.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        self.mock_get_cache.return_value = self.mock_cache

        self.graph = WikidataGraph()

    def teardown_method(self) -> None:
        self.cache_patcher.stop()
        self.sparql_patcher.stop()

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_resolve_qid_labels_empty(self, mock_get: MagicMock) -> None:
        result = self.graph._resolve_qid_labels(set())
        assert result == {}
        mock_get.assert_not_called()

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_resolve_qid_labels_non_200(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp
        result = self.graph._resolve_qid_labels({"Q243"})
        assert result == {}

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_resolve_qid_labels_network_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = ConnectionError("network failure")
        result = self.graph._resolve_qid_labels({"Q243"})
        assert result == {}

    @patch("watson_lite.graph.wikidata.requests.get")
    def test_enrich_cleaned_name_empty(self, mock_get: MagicMock) -> None:
        with patch.object(self.graph, "find_entity_id", return_value=None):
            result = self.graph.enrich("")
            assert result is not None
            assert result.wikidata_id is None


# ---------------------------------------------------------------------------
# Pipeline coverage — _log_graph_results with real GraphResult data
# ---------------------------------------------------------------------------


class TestPipelineCoverage:
    def setup_method(self) -> None:
        self.patches = [
            patch("watson_lite.pipeline.NLPProcessor"),
            patch("watson_lite.pipeline.BM25Retriever"),
            patch("watson_lite.pipeline.VectorRetriever"),
            patch("watson_lite.pipeline.WikidataGraph"),
            patch("watson_lite.pipeline.Ranker"),
            patch("watson_lite.pipeline.ExtractiveReader"),
            patch("watson_lite.pipeline.ConfidenceScorer"),
        ]
        self.mocks = [p.start() for p in self.patches]
        (
            self.mock_nlp_cls,
            self.mock_bm25_cls,
            self.mock_vector_cls,
            self.mock_graph_cls,
            self.mock_ranker_cls,
            self.mock_reader_cls,
            self.mock_scorer_cls,
        ) = self.mocks

        self.mock_nlp = MagicMock()
        self.mock_nlp_cls.return_value = self.mock_nlp
        self.mock_bm25 = MagicMock()
        self.mock_bm25_cls.return_value = self.mock_bm25
        self.mock_vector = MagicMock()
        self.mock_vector_cls.return_value = self.mock_vector
        self.mock_graph = MagicMock()
        self.mock_graph_cls.return_value = self.mock_graph
        self.mock_ranker = MagicMock()
        self.mock_ranker_cls.return_value = self.mock_ranker
        self.mock_reader = MagicMock()
        self.mock_reader_cls.return_value = self.mock_reader
        self.mock_scorer = MagicMock()
        self.mock_scorer_cls.return_value = self.mock_scorer

        self.fetch_patcher = patch(
            "watson_lite.retrieval.dataset_plugins.fetch_wikipedia_passages"
        )
        self.mock_fetch = self.fetch_patcher.start()

        from watson_lite.pipeline import WatsonLite

        self.pipeline = WatsonLite()

    def teardown_method(self) -> None:
        self.fetch_patcher.stop()
        for p in self.patches:
            p.stop()

    def test_log_graph_results_with_data(self) -> None:
        from watson_lite.core.models import ParsedQuestion

        self.mock_fetch.return_value = [Passage("test", "s", "u")]
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test",
            question_type="test",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 13}],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test"],
            keywords=[],
        )
        self.mock_graph.enrich_all.return_value = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="architect",
                        value="Gustave Eiffel",
                    )
                ],
            )
        ]
        self.mock_reader.extract.return_value = [
            AnswerCandidate(
                span="ans", source="s", url="", passage="", extraction_score=0.9, rank=1
            )
        ]
        self.mock_scorer.score.return_value = FinalAnswer(
            answer="ans", confidence=0.5, source="s", url=""
        )
        result = self.pipeline.answer("test question", verbose=True)
        assert result.answer == "ans"


# ---------------------------------------------------------------------------
# Cache — thread safety, SQL injection via keys, unicode keys
# ---------------------------------------------------------------------------


class TestCacheAdversarialExtended:
    def setup_method(self) -> None:
        fd, self.tmp = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        self.cache = Cache(self.tmp)

    def teardown_method(self) -> None:
        self.cache.close()
        os.unlink(self.tmp)

    def test_sql_injection_via_key(self) -> None:
        malicious_key = "'; DROP TABLE cache; --"
        self.cache.set(malicious_key, "exfiltrated")
        assert self.cache.get(malicious_key) == "exfiltrated"
        self.cache.set("other", "data")
        assert self.cache.get("other") == "data"

    def test_unicode_key(self) -> None:
        self.cache.set("café", "unicafe")
        assert self.cache.get("café") == "unicafe"

    def test_emoji_key_and_value(self) -> None:
        self.cache.set("emoji", "🚀✨🔥")
        assert self.cache.get("emoji") == "🚀✨🔥"

    def test_binary_value(self) -> None:
        self.cache.set("bin", b"\x00\x01\x02")
        val = self.cache.get("bin")
        assert isinstance(val, str)

    def test_nested_data_structures(self) -> None:
        data = {
            "list": [1, 2, {"a": [3, 4]}],
            "tuple": (5, 6),
            "bool": True,
            "none": None,
            "float": 3.14,
        }
        self.cache.set("nested", data)
        result = self.cache.get("nested")
        assert result["list"] == [1, 2, {"a": [3, 4]}]
        assert result["bool"] is True
        assert result["none"] is None
        assert result["float"] == 3.14

    def test_concurrent_read_write_does_not_corrupt(self) -> None:
        import threading

        errors = []

        def writer() -> None:
            for i in range(50):
                try:
                    self.cache.set(f"k{i}", f"v{i}")
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            for i in range(50):
                try:
                    self.cache.get(f"k{i}")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_get_on_empty_db(self) -> None:
        fd, tmp = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        c = Cache(tmp)
        assert c.get("anything") is None
        assert c.get_or_sentinel("anything") is SENTINEL
        c.close()
        os.unlink(tmp)

    def test_overwrite_with_none(self) -> None:
        self.cache.set("x", "original")
        self.cache.set("x", None)
        assert self.cache.get("x") is None
        assert self.cache.get_or_sentinel("x") is None

    def test_clear_then_get(self) -> None:
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        self.cache.clear()
        assert self.cache.get("a") is None
        assert self.cache.get("b") is None
        self.cache.set("a", 3)
        assert self.cache.get("a") == 3


# ---------------------------------------------------------------------------
# NLP — fuzzing-level adversarial inputs
# ---------------------------------------------------------------------------


class TestNLPFuzzing:
    def test_classify_question_with_control_chars(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        for q in ["\x00\x01\x02", "who\x00ami", "what\x1b[3J"]:
            parsed = nlp.process(q)
            assert isinstance(parsed.question_type, str)

    def test_classify_question_mixed_scripts(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        parsed = nlp.process("Who built 塔?")
        assert parsed.question_type == "who"
        assert len(parsed.entities) >= 0

    def test_classify_question_numbers_only(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        assert nlp.classify_question("12345") == "unknown"
        assert nlp.classify_question("42") == "unknown"

    def test_process_question_with_only_stop_words(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        parsed = nlp.process("the and of in a to is")
        assert parsed.keywords == []

    def test_extract_keywords_from_empty_doc(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        doc = nlp.nlp("...")
        kws = nlp.extract_keywords(doc)
        assert isinstance(kws, list)

    def test_get_root_verb_no_verb(self) -> None:
        from watson_lite.core.nlp import NLPProcessor

        nlp = NLPProcessor()
        doc = nlp.nlp("The beautiful red car.")
        assert nlp.get_root_verb(doc) is None


# ---------------------------------------------------------------------------
# BM25 — degenerate inputs
# ---------------------------------------------------------------------------


class TestBM25Degenerate:
    def setup_method(self) -> None:
        self.cache_patcher = patch("watson_lite.retrieval.bm25_retriever.get_cache")
        self.mock_get_cache = self.cache_patcher.start()
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        self.mock_get_cache.return_value = self.mock_cache

        self.bm25s_patcher = patch("watson_lite.retrieval.bm25_retriever.bm25s")
        self.mock_bm25s = self.bm25s_patcher.start()
        self.mock_bm25s.tokenize.return_value = "tokenized_corpus"
        self.mock_retriever_instance = MagicMock()
        self.mock_bm25s.BM25.return_value = self.mock_retriever_instance

        self.retriever = BM25Retriever()

    def teardown_method(self) -> None:
        self.cache_patcher.stop()
        self.bm25s_patcher.stop()

    def test_index_and_retrieve_single_passage(self) -> None:
        self.retriever.index([Passage("unique text here", "s", "u")])
        self.mock_retriever_instance.retrieve.return_value = (
            [["unique text here"]],
            [[0.5]],
        )
        result = self.retriever.retrieve("query", top_k=10)
        assert len(result) == 1

    def test_retrieve_with_no_index(self) -> None:
        result = self.retriever.retrieve("query", top_k=10)
        assert result == []

    def test_index_twice_overwrites(self) -> None:
        p1 = Passage("first corpus", "s", "u")
        p2 = Passage("second corpus", "s", "u")
        self.retriever.index([p1])
        self.retriever.index([p2])
        self.mock_retriever_instance.retrieve.return_value = (
            [["second corpus"]],
            [[0.9]],
        )
        result = self.retriever.retrieve("query", top_k=10)
        assert len(result) == 1

    def test_cache_returning_none_for_passages(self) -> None:
        self.mock_cache.get_or_sentinel.return_value = SENTINEL
        from watson_lite.retrieval.bm25_retriever import fetch_wikipedia_passages

        with (
            patch("watson_lite.retrieval.bm25_retriever.requests") as mock_requests,
            patch(
                "watson_lite.retrieval.bm25_retriever.ThreadPoolExecutor"
            ) as mock_exec,
        ):
            search_resp = MagicMock()
            search_resp.status_code = 200
            search_resp.json.return_value = {"query": {"search": [{"title": "Test"}]}}
            mock_requests.get.return_value = search_resp

            mock_future = MagicMock()
            mock_future.result.return_value = [
                Passage(
                    "hello world " * 50, "Test", "https://en.wikipedia.org/wiki/Test"
                )
            ]
            mock_executor = MagicMock()
            mock_executor.__enter__.return_value = mock_executor
            mock_executor.submit.return_value = mock_future
            mock_exec.return_value = mock_executor

            with patch(
                "watson_lite.retrieval.bm25_retriever.as_completed",
                return_value=[mock_future],
            ):
                result = fetch_wikipedia_passages("test")
            assert len(result) > 0
            assert result[0].text is not None


# ---------------------------------------------------------------------------
# Vector retriever — degenerate inputs
# ---------------------------------------------------------------------------


class TestVectorDegenerate:
    def setup_method(self) -> None:
        self.faiss_patcher = patch("watson_lite.retrieval.vector_retriever.faiss")
        self.mock_faiss = self.faiss_patcher.start()
        self.mock_index = MagicMock()
        self.mock_index.ntotal = 0
        self.mock_faiss.IndexFlatIP.return_value = self.mock_index

        self.model_patcher = patch(
            "watson_lite.retrieval.vector_retriever.SentenceTransformer"
        )
        self.mock_model_cls = self.model_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model_cls.return_value = self.mock_model
        self.mock_model.get_sentence_embedding_dimension.return_value = 384

        self.retriever = VectorRetriever()

    def teardown_method(self) -> None:
        self.faiss_patcher.stop()
        self.model_patcher.stop()

    def test_retrieve_without_indexing(self) -> None:
        result = self.retriever.retrieve("query", top_k=10)
        assert result == []

    def test_index_zero_passages(self) -> None:
        self.mock_model.encode.return_value = np.empty((0, 384), dtype="float32")
        self.retriever.index_passages([])
        assert len(self.retriever.passages) == 0

    def test_all_identical_passages(self) -> None:
        texts = ["same text"] * 5
        passages = [Passage(t, "s", "u") for t in texts]
        self.mock_model.encode.return_value = np.ones((5, 384), dtype="float32")
        self.retriever.index_passages(passages)
        self.mock_index.ntotal = 5
        self.mock_index.search.return_value = (
            np.array([[0.9] * 3]),
            np.array([[0, 1, 2]]),
        )
        results = self.retriever.retrieve("query", top_k=3)
        assert len(results) == 3
        for r in results:
            assert r.text == "same text"


# ---------------------------------------------------------------------------
# RRF — pathological rank inputs
# ---------------------------------------------------------------------------


class TestRRFAdversarial:
    def test_rrf_with_negative_k(self) -> None:
        ps = [Passage("text", "s", "u", rank=1)]
        with pytest.raises(ZeroDivisionError):
            RRFFusion().fuse([ps], k=-1)

    def test_rrf_with_zero_k(self) -> None:
        ps = [Passage("text", "s", "u", rank=1)]
        result = RRFFusion().fuse([ps], k=0)
        assert len(result) == 1

    def test_rrf_with_duplicate_in_same_list(self) -> None:
        p1 = Passage("same", "s", "u")
        result = RRFFusion().fuse([[p1, p1]], k=60)
        assert len(result) == 1

    def test_rrf_with_many_lists(self) -> None:
        ps = [Passage(f"text{i}", "s", "u") for i in range(3)]
        result = RRFFusion().fuse([ps, ps, ps, ps, ps], k=60)
        assert len(result) == 3

    def test_rrf_rank_assignment(self) -> None:
        p1 = Passage("a", "s", "u")
        p2 = Passage("b", "s", "u")
        result = RRFFusion().fuse([[p1, p2], [p2, p1]], k=60)
        for r in result:
            assert r.rank in (1, 2)


# ---------------------------------------------------------------------------
# Confidence scorer — pathological inputs
# ---------------------------------------------------------------------------


class TestConfidenceScorerPathological:
    def test_span_empty_string(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="", source="s", url="", passage="", extraction_score=0.9, rank=1
            ),
        ]
        answer = scorer.score(candidates, [], "what")
        assert answer.confidence > 0

    def test_very_long_span(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="x" * 1000,
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        answer = scorer.score(candidates, [], "what")
        assert answer.confidence > 0

    def test_all_candidates_identical_span(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="s",
                url="",
                passage=f"p{i}",
                extraction_score=0.8,
                rank=i + 1,
            )
            for i in range(10)
        ]
        answer = scorer.score(candidates, [], "who")
        assert answer.confidence_breakdown["span_agreement"] == 1.0

    def test_rank_signals_for_all_positions(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="x", source="s", url="", passage="", extraction_score=0.9, rank=r
            )
            for r in range(1, 12)
        ]
        answer = scorer.score(candidates, [], "what")
        breakdown = answer.confidence_breakdown
        assert breakdown["span_agreement"] > 0


# ---------------------------------------------------------------------------
# Pipeline — component failure propagation
# ---------------------------------------------------------------------------


class TestPipelineFailure:
    def setup_method(self) -> None:
        self.patches = [
            patch("watson_lite.pipeline.NLPProcessor"),
            patch("watson_lite.pipeline.BM25Retriever"),
            patch("watson_lite.pipeline.VectorRetriever"),
            patch("watson_lite.pipeline.WikidataGraph"),
            patch("watson_lite.pipeline.Ranker"),
            patch("watson_lite.pipeline.ExtractiveReader"),
            patch("watson_lite.pipeline.ConfidenceScorer"),
        ]
        self.mocks = [p.start() for p in self.patches]
        (
            self.mock_nlp_cls,
            self.mock_bm25_cls,
            self.mock_vector_cls,
            self.mock_graph_cls,
            self.mock_ranker_cls,
            self.mock_reader_cls,
            self.mock_scorer_cls,
        ) = self.mocks

        self.mock_nlp = MagicMock()
        self.mock_nlp_cls.return_value = self.mock_nlp
        self.mock_bm25 = MagicMock()
        self.mock_bm25_cls.return_value = self.mock_bm25
        self.mock_vector = MagicMock()
        self.mock_vector_cls.return_value = self.mock_vector
        self.mock_graph = MagicMock()
        self.mock_graph_cls.return_value = self.mock_graph
        self.mock_ranker = MagicMock()
        self.mock_ranker_cls.return_value = self.mock_ranker
        self.mock_reader = MagicMock()
        self.mock_reader_cls.return_value = self.mock_reader
        self.mock_scorer = MagicMock()
        self.mock_scorer_cls.return_value = self.mock_scorer

        self.fetch_patcher = patch(
            "watson_lite.retrieval.dataset_plugins.fetch_wikipedia_passages"
        )
        self.mock_fetch = self.fetch_patcher.start()

        from watson_lite.pipeline import WatsonLite

        self.pipeline = WatsonLite()

    def teardown_method(self) -> None:
        self.fetch_patcher.stop()
        for p in self.patches:
            p.stop()

    def test_graph_enrich_raises_exception(self) -> None:
        from watson_lite.core.models import ParsedQuestion

        self.mock_fetch.return_value = [Passage("test", "s", "u")]
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test",
            question_type="test",
            entities=[{"text": "entity", "label": "ORG", "start": 0, "end": 6}],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test"],
            keywords=[],
        )
        self.mock_graph.enrich_all.side_effect = RuntimeError("Wikidata down")
        with pytest.raises(RuntimeError, match="Wikidata down"):
            self.pipeline.answer("test")

    def test_reader_returns_no_candidates(self) -> None:
        from watson_lite.core.models import ParsedQuestion

        self.mock_fetch.return_value = [Passage("test", "s", "u")]
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test",
            question_type="test",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test"],
            keywords=[],
        )
        self.mock_reader.extract.return_value = []
        mock_answer = FinalAnswer(
            answer="No answer found",
            confidence=0.0,
            source="",
            url="",
            confidence_breakdown={"reason": "no candidates"},
        )
        self.mock_scorer.score.return_value = mock_answer
        result = self.pipeline.answer("test")
        assert result.answer == "No answer found"

    def test_retrieval_returns_empty_lists(self) -> None:
        from watson_lite.core.models import ParsedQuestion

        self.mock_fetch.return_value = [Passage("test", "s", "u")]
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test",
            question_type="test",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test"],
            keywords=[],
        )
        self.mock_reader.extract.return_value = []
        self.mock_scorer.score.return_value = FinalAnswer(
            answer="nothing", confidence=0.0, source="", url=""
        )
        result = self.pipeline.answer("test")
        assert result is not None


# ---------------------------------------------------------------------------
# End-to-end — network failure scenarios
# ---------------------------------------------------------------------------


class TestE2EFailures:
    def test_e2e_wikidata_entity_search_429(self) -> None:
        from watson_lite.pipeline import WatsonLite

        wl = object.__new__(WatsonLite)
        from watson_lite.core.config import FeatureConfig

        wl.config = FeatureConfig.baseline()
        wl.nlp = MagicMock()
        wl.bm25 = MagicMock()
        wl.vector = MagicMock()
        wl.graph = MagicMock()
        wl.ranker = MagicMock()
        wl.reader = MagicMock()
        wl.scorer = MagicMock()
        wl.dataset_query_engine = MagicMock()
        wl._passage_cache = {}
        wl._index_loaded = False
        wl.logger = MagicMock()
        test_passages = [
            Passage(
                "Eiffel Tower was designed by Gustave Eiffel.", "Eiffel Tower", "url"
            )
        ]
        wl.dataset_query_engine.query.return_value = test_passages
        wl.bm25.retrieve.return_value = test_passages
        wl.vector.retrieve.return_value = test_passages

        wl.nlp.process.return_value = MagicMock(
            sub_questions=["Who designed the Eiffel Tower?"],
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 13}],
            question_type="who",
            noun_chunks=[],
            root_verb=None,
            keywords=[],
        )
        wl.scorer.score.return_value = FinalAnswer(
            answer="Gustave Eiffel", confidence=0.5, source="", url=""
        )
        wl.reader.extract.return_value = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            )
        ]
        result = wl.answer("Who designed the Eiffel Tower?", verbose=False)
        assert result.answer == "Gustave Eiffel"


# ---------------------------------------------------------------------------
# Term match scoring — IDF-weighted passage term overlap
# ---------------------------------------------------------------------------


class TestTermMatch:
    def test_perfect_match_returns_high_score(self) -> None:
        from watson_lite.scoring.term_match import score_term_match

        passages = [
            RankedPassage(
                passage=Passage(
                    "Gustave Eiffel designed the Eiffel Tower",
                    "src",
                    "url",
                ),
                rank=1,
            ),
        ]
        score = score_term_match("Gustave Eiffel designed the Eiffel Tower", passages)
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self) -> None:
        from watson_lite.scoring.term_match import score_term_match

        passages = [
            RankedPassage(
                passage=Passage(
                    "The weather is nice today",
                    "src",
                    "url",
                ),
                rank=1,
            ),
        ]
        score = score_term_match("Gustave Eiffel designed the Eiffel Tower", passages)
        assert score == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        from watson_lite.scoring.term_match import score_term_match

        passages = [
            RankedPassage(
                passage=Passage(
                    "Paris is the capital of France",
                    "src",
                    "url",
                ),
                rank=1,
            ),
        ]
        score = score_term_match("Eiffel Tower Paris", passages)
        assert 0.0 < score < 1.0

    def test_empty_passages_returns_zero(self) -> None:
        from watson_lite.scoring.term_match import score_term_match

        assert score_term_match("test", []) == 0.0

    def test_empty_question_returns_zero(self) -> None:
        from watson_lite.scoring.term_match import score_term_match

        passages = [
            RankedPassage(passage=Passage("hello world", "src", "url"), rank=1),
        ]
        assert score_term_match("", passages) == 0.0

    def test_rare_terms_get_higher_weight(self) -> None:
        from watson_lite.scoring.term_match import score_term_match

        passages = [
            RankedPassage(
                passage=Passage("Gustave Eiffel built the tower", "src", "url"),
                rank=1,
            ),
            RankedPassage(
                passage=Passage("the tower is tall", "src", "url"),
                rank=2,
            ),
        ]
        score = score_term_match("Gustave Eiffel tower", passages)
        assert score > 0.5

    def test_selects_best_passage(self) -> None:
        from watson_lite.scoring.term_match import score_term_match

        passages = [
            RankedPassage(
                passage=Passage("the and of in a", "src", "url"),
                rank=1,
            ),
            RankedPassage(
                passage=Passage("Gustave Eiffel designed it", "src", "url"),
                rank=2,
            ),
        ]
        score = score_term_match("Gustave Eiffel", passages)
        assert score > 0.5

    def test_tokenization_handles_contractions(self) -> None:
        from watson_lite.scoring.term_match import _tokenize

        tokens = _tokenize("Don't forget")
        assert "don" in tokens
        assert "t" in tokens
        assert "forget" in tokens

    def test_tokenization_lowercases_and_filters_stops(self) -> None:
        from watson_lite.scoring.term_match import _tokenize

        tokens = _tokenize("Hello WORLD the and")
        assert "hello" in tokens
        assert "world" in tokens
        assert "the" not in tokens
        assert "and" not in tokens

    def test_integrates_with_confidence_scorer(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="s",
                url="",
                passage="Gustave Eiffel designed the Eiffel Tower",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        ranked_passages = [
            RankedPassage(
                passage=Passage(
                    "Gustave Eiffel designed the Eiffel Tower",
                    "s",
                    "url",
                ),
                rank=1,
            ),
        ]
        answer = scorer.score(
            candidates,
            [],
            "who",
            question="Who designed the Eiffel Tower?",
            ranked_passages=ranked_passages,
        )
        assert answer.confidence_breakdown["term_match"] > 0

    def test_disabled_via_flag(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="x",
                source="s",
                url="",
                passage="x y z",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        ranked_passages = [
            RankedPassage(passage=Passage("x y z", "s", "url"), rank=1),
        ]
        answer = scorer.score(
            candidates,
            [],
            "what",
            question="x y z",
            ranked_passages=ranked_passages,
            enable_term_match=False,
        )
        assert answer.confidence_breakdown["term_match"] == 0.0


# ---------------------------------------------------------------------------
# Temporal & geospatial consistency scoring
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_temporal_match_from_graph(self) -> None:
        from watson_lite.scoring.consistency import score_temporal_consistency

        candidates = [
            AnswerCandidate(
                span="1889",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="inception",
                        value="1889",
                    ),
                ],
            )
        ]
        score = score_temporal_consistency(candidates, graph_results)
        assert score == 1.0

    def test_temporal_no_match_returns_zero(self) -> None:
        from watson_lite.scoring.consistency import score_temporal_consistency

        candidates = [
            AnswerCandidate(
                span="1800",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="inception",
                        value="1889",
                    ),
                ],
            )
        ]
        score = score_temporal_consistency(candidates, graph_results)
        assert score == 0.0

    def test_temporal_no_year_in_span_returns_zero(self) -> None:
        from watson_lite.scoring.consistency import score_temporal_consistency

        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="inception",
                        value="1889",
                    ),
                ],
            )
        ]
        score = score_temporal_consistency(candidates, graph_results)
        assert score == 0.0

    def test_temporal_with_no_temporal_facts_returns_zero(self) -> None:
        from watson_lite.scoring.consistency import score_temporal_consistency

        candidates = [
            AnswerCandidate(
                span="1889",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="architect",
                        value="Gustave Eiffel",
                    ),
                ],
            )
        ]
        score = score_temporal_consistency(candidates, graph_results)
        assert score == 0.0

    def test_temporal_empty_candidates_returns_zero(self) -> None:
        from watson_lite.scoring.consistency import score_temporal_consistency

        assert score_temporal_consistency([], []) == 0.0

    def test_geo_match_from_graph(self) -> None:
        from watson_lite.scoring.consistency import score_geospatial_consistency

        candidates = [
            AnswerCandidate(
                span="France",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="country",
                        value="France",
                    ),
                ],
            )
        ]
        score = score_geospatial_consistency(candidates, graph_results)
        assert score == pytest.approx(1.0)

    def test_geo_no_match_returns_zero(self) -> None:
        from watson_lite.scoring.consistency import score_geospatial_consistency

        candidates = [
            AnswerCandidate(
                span="Germany",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="country",
                        value="France",
                    ),
                ],
            )
        ]
        score = score_geospatial_consistency(candidates, graph_results)
        assert score == 0.0

    def test_geo_empty_candidates_returns_zero(self) -> None:
        from watson_lite.scoring.consistency import score_geospatial_consistency

        assert score_geospatial_consistency([], []) == 0.0

    def test_geo_case_insensitive_match(self) -> None:
        from watson_lite.scoring.consistency import score_geospatial_consistency

        candidates = [
            AnswerCandidate(
                span="france",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="country",
                        value="France",
                    ),
                ],
            )
        ]
        score = score_geospatial_consistency(candidates, graph_results)
        assert score == pytest.approx(1.0)

    def test_integrates_with_confidence_scorer(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="France",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="country",
                        value="France",
                    ),
                ],
            )
        ]
        answer = scorer.score(candidates, graph_results, "where")
        assert answer.confidence_breakdown["geospatial_consistency"] > 0
        assert "temporal_consistency" in answer.confidence_breakdown

    def test_disabled_via_flag(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="France",
                source="s",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        graph_results = [
            GraphResult(
                entity_name="Eiffel Tower",
                wikidata_id="Q243",
                facts=[
                    EntityFact(
                        entity="Q243",
                        property_label="country",
                        value="France",
                    ),
                ],
            )
        ]
        answer = scorer.score(
            candidates,
            graph_results,
            "where",
            enable_consistency=False,
        )
        assert answer.confidence_breakdown["geospatial_consistency"] == 0.0
        assert answer.confidence_breakdown["temporal_consistency"] == 0.0


# ---------------------------------------------------------------------------
# Answer merging — Wikidata QID-based candidate deduplication
# ---------------------------------------------------------------------------


class TestAnswerMerging:
    def test_merge_identical_spans_does_nothing(self) -> None:
        from watson_lite.scoring.answer_merging import merge_candidates_by_qid

        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            return_value=None,
        ):
            candidates = [
                AnswerCandidate(
                    span="Paris",
                    source="a",
                    url="",
                    passage="",
                    extraction_score=0.9,
                    rank=1,
                ),
                AnswerCandidate(
                    span="Paris",
                    source="b",
                    url="",
                    passage="",
                    extraction_score=0.8,
                    rank=2,
                ),
            ]
            result = merge_candidates_by_qid(candidates)
            assert len(result) == 2

    def test_merge_different_spans_same_qid(self) -> None:
        from watson_lite.scoring.answer_merging import merge_candidates_by_qid

        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            side_effect=lambda s: "Q243" if "Eiffel" in s else None,
        ):
            candidates = [
                AnswerCandidate(
                    span="Gustave Eiffel",
                    source="a",
                    url="",
                    passage="",
                    extraction_score=0.9,
                    rank=1,
                ),
                AnswerCandidate(
                    span="Alexandre Gustave Eiffel",
                    source="b",
                    url="",
                    passage="",
                    extraction_score=0.8,
                    rank=2,
                ),
            ]
            result = merge_candidates_by_qid(candidates)
            assert len(result) == 1
            assert result[0].span == "Gustave Eiffel"
            assert result[0].extraction_score == 0.9

    def test_merge_preserves_best_rank(self) -> None:
        from watson_lite.scoring.answer_merging import merge_candidates_by_qid

        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            return_value="Q243",
        ):
            candidates = [
                AnswerCandidate(
                    span="Eiffel",
                    source="a",
                    url="",
                    passage="",
                    extraction_score=0.7,
                    rank=10,
                ),
                AnswerCandidate(
                    span="Gustave Eiffel",
                    source="b",
                    url="",
                    passage="",
                    extraction_score=0.9,
                    rank=1,
                ),
            ]
            result = merge_candidates_by_qid(candidates)
            assert len(result) == 1
            assert result[0].rank == 1

    def test_merge_different_qids_keeps_separate(self) -> None:
        from watson_lite.scoring.answer_merging import merge_candidates_by_qid

        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            side_effect=lambda s: {"Paris": "Q90", "London": "Q84"}.get(s),
        ):
            candidates = [
                AnswerCandidate(
                    span="Paris",
                    source="a",
                    url="",
                    passage="",
                    extraction_score=0.9,
                    rank=1,
                ),
                AnswerCandidate(
                    span="London",
                    source="b",
                    url="",
                    passage="",
                    extraction_score=0.8,
                    rank=2,
                ),
            ]
            result = merge_candidates_by_qid(candidates)
            assert len(result) == 2

    def test_merge_empty_list(self) -> None:
        from watson_lite.scoring.answer_merging import merge_candidates_by_qid

        assert merge_candidates_by_qid([]) == []

    def test_merge_single_candidate(self) -> None:
        from watson_lite.scoring.answer_merging import merge_candidates_by_qid

        c = AnswerCandidate(
            span="Paris",
            source="a",
            url="",
            passage="",
            extraction_score=0.9,
            rank=1,
        )
        result = merge_candidates_by_qid([c])
        assert len(result) == 1
        assert result[0] is c

    def test_merge_integrates_with_confidence_scorer(self) -> None:
        from watson_lite.scoring.answer_merging import merge_candidates_by_qid

        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="a",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
            AnswerCandidate(
                span="Gustave Eiffel",
                source="b",
                url="",
                passage="",
                extraction_score=0.8,
                rank=2,
            ),
        ]
        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            return_value=None,
        ):
            answer = scorer.score(candidates, [], "who")
            assert answer.answer == "Gustave Eiffel"
            assert "answer_merging" not in answer.confidence_breakdown

    def test_disabled_via_flag(self) -> None:
        scorer = ConfidenceScorer()
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="a",
                url="",
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
            AnswerCandidate(
                span="Paris",
                source="b",
                url="",
                passage="",
                extraction_score=0.8,
                rank=2,
            ),
        ]
        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            side_effect=AssertionError("should not be called"),
        ):
            answer = scorer.score(candidates, [], "where", enable_answer_merging=False)
            assert answer.answer == "Paris"
