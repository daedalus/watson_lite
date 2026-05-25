from unittest.mock import MagicMock, patch

import pytest

from watson_lite.core.config import FeatureConfig
from watson_lite.core.models import (
    AnswerCandidate,
    FinalAnswer,
    ParsedQuestion,
    Passage,
    RankedPassage,
)
from watson_lite.pipeline import WatsonLite


class TestWatsonLite:
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

        self.mock_nlp_cls, self.mock_bm25_cls, self.mock_vector_cls = (
            self.mocks[0],
            self.mocks[1],
            self.mocks[2],
        )
        self.mock_graph_cls, self.mock_ranker_cls = (
            self.mocks[3],
            self.mocks[4],
        )
        self.mock_reader_cls, self.mock_scorer_cls = (
            self.mocks[5],
            self.mocks[6],
        )

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

        self.fetch_patcher = patch("watson_lite.pipeline.fetch_wikipedia_passages")
        self.mock_fetch = self.fetch_patcher.start()
        self.fetch_page_patcher = patch(
            "watson_lite.pipeline.fetch_wikipedia_page_by_title"
        )
        self.mock_fetch_page = self.fetch_page_patcher.start()
        self.mock_fetch_page.return_value = []

        self.base_config = (
            FeatureConfig.baseline()
            .with_feature("multi_hypothesis", False)
            .with_feature("per_candidate_retrieval", False)
            .with_feature("bidirectional_validation", False)
            .with_feature("iterative_retrieval", False)
        )
        self.pipeline = WatsonLite(config=self.base_config)

    def _setup_success_flow(self) -> None:
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test question",
            question_type="what",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 5}],
            noun_chunks=["Eiffel Tower"],
            root_verb="design",
            sub_questions=["test question"],
            keywords=["test", "question"],
            lat_qids=["Q5"],
        )
        passages = [
            Passage(
                text="Gustave Eiffel designed the tower.",
                source="Wikipedia",
                url="http://example.com",
            )
        ]
        self.mock_fetch.return_value = passages
        self.mock_bm25.retrieve.return_value = passages
        self.mock_vector.retrieve.return_value = passages
        self.mock_graph.enrich_all.return_value = []
        self.mock_ranker.rank.return_value = [
            RankedPassage(
                passage=passages[0],
                rrf_score=0.5,
                cross_score=0.9,
                final_score=0.9,
                rank=1,
            )
        ]
        self.mock_reader.extract.return_value = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="Wikipedia",
                url="http://example.com",
                passage="Gustave Eiffel designed the tower.",
                extraction_score=0.95,
                rank=1,
            )
        ]
        self.mock_scorer.score.return_value = FinalAnswer(
            answer="Gustave Eiffel",
            confidence=0.9,
            source="Wikipedia",
            url="http://example.com",
        )

    def teardown_method(self) -> None:
        for p in self.patches:
            p.stop()
        self.fetch_patcher.stop()
        self.fetch_page_patcher.stop()

    def test_constructor_initializes_all_components(self) -> None:
        self.mock_bm25_cls.assert_called_once()
        self.mock_scorer_cls.assert_called_once()
        self.mock_nlp_cls.assert_not_called()
        self.mock_vector_cls.assert_not_called()
        self.mock_graph_cls.assert_not_called()
        self.mock_ranker_cls.assert_not_called()
        self.mock_reader_cls.assert_not_called()
        assert self.pipeline.nlp is None
        assert self.pipeline.vector is None
        assert self.pipeline.graph is None
        assert self.pipeline.ranker is None
        assert self.pipeline.reader is None
        assert self.pipeline._last_passage_hash is None

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            self.pipeline.answer("")

    def test_answer_no_passages(self) -> None:
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test",
            question_type="what",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test"],
            keywords=["test"],
        )
        self.mock_fetch.return_value = []

        result = self.pipeline.answer("test question")
        assert result.answer == "Could not retrieve relevant passages."
        assert result.confidence == 0.0

    def test_answer_full_success(self) -> None:
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="Who designed the Eiffel Tower?",
            question_type="who",
            entities=[{"text": "Eiffel Tower", "label": "ORG", "start": 0, "end": 5}],
            noun_chunks=["the Eiffel Tower"],
            root_verb="design",
            sub_questions=["Who designed the Eiffel Tower?"],
            keywords=["design", "Eiffel", "Tower"],
        )

        test_passages = [
            Passage(
                text="Gustave Eiffel designed the tower.",
                source="Wikipedia",
                url="http://example.com",
            )
        ]
        self.mock_fetch.return_value = test_passages

        self.mock_bm25.retrieve.return_value = test_passages
        self.mock_vector.retrieve.return_value = test_passages
        self.mock_graph.enrich_all.return_value = []

        ranked = [
            RankedPassage(
                passage=test_passages[0],
                rrf_score=0.5,
                cross_score=0.9,
                final_score=0.9,
                rank=1,
            )
        ]
        self.mock_ranker.rank.return_value = ranked

        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="Wikipedia",
                url="http://example.com",
                passage="Gustave Eiffel designed the tower.",
                extraction_score=0.95,
                rank=1,
            )
        ]
        self.mock_reader.extract.return_value = candidates

        expected_answer = FinalAnswer(
            answer="Gustave Eiffel",
            confidence=0.85,
            source="Wikipedia",
            url="http://example.com",
            supporting_passages=["Gustave Eiffel designed the tower."],
            graph_facts=["architect: Gustave Eiffel"],
            confidence_breakdown={
                "extraction_model": 0.95,
                "span_agreement": 0.333,
                "graph_corroboration": 0.2,
                "passage_rank_signal": 1.0,
            },
        )
        self.mock_scorer.score.return_value = expected_answer

        result = self.pipeline.answer("Who designed the Eiffel Tower?")

        assert result.answer == "Gustave Eiffel"
        assert result.confidence == 0.85
        self.mock_nlp.process.assert_called_once()
        self.mock_bm25.index.assert_called_once_with(test_passages)
        self.mock_bm25.retrieve.assert_called_once()
        self.mock_vector.index_passages.assert_called_once_with(test_passages)
        self.mock_vector.retrieve.assert_called_once()
        self.mock_graph.enrich_all.assert_called_once()
        self.mock_ranker.rank.assert_called_once()
        self.mock_reader.extract.assert_called_once()
        self.mock_scorer.score.assert_called_once()

    def test_answer_verbose_false(self) -> None:
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="test",
            question_type="what",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test"],
            keywords=["test"],
        )
        self.mock_fetch.return_value = [
            Passage(
                text="test text",
                source="Wiki",
                url="http://e.com",
            )
        ]
        self.mock_bm25.retrieve.return_value = []
        self.mock_vector.retrieve.return_value = []
        self.mock_ranker.rank.return_value = []
        self.mock_reader.extract.return_value = []
        self.mock_scorer.score.return_value = FinalAnswer(
            answer="No answer found",
            confidence=0.0,
            source="",
            url="",
        )

        result = self.pipeline.answer("test", verbose=False)
        assert result.answer == "No answer found"

    def test_reindex_skipped_on_same_passages(self) -> None:
        """index/index_passages must be called only once when passages are identical."""
        parsed = ParsedQuestion(
            raw="test",
            question_type="what",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["test"],
            keywords=["test"],
        )
        passages = [Passage(text="same text", source="Wiki", url="http://e.com")]
        answer_obj = FinalAnswer(
            answer="ans", confidence=0.9, source="Wiki", url="http://e.com"
        )

        self.mock_nlp.process.return_value = parsed
        self.mock_fetch.return_value = passages
        self.mock_bm25.retrieve.return_value = []
        self.mock_vector.retrieve.return_value = []
        self.mock_ranker.rank.return_value = []
        self.mock_reader.extract.return_value = []
        self.mock_scorer.score.return_value = answer_obj

        # First call should trigger indexing for both retrievers.
        self.pipeline.answer("test", verbose=False)
        assert self.mock_bm25.index.call_count == 1
        assert self.mock_vector.index_passages.call_count == 1

        # Second call reuses the same passages and should skip re-indexing.
        self.pipeline.answer("test", verbose=False)
        assert self.mock_bm25.index.call_count == 1
        assert self.mock_vector.index_passages.call_count == 1

    def test_reindex_on_different_passages(self) -> None:
        """index/index_passages must be called again when passages change."""
        parsed_1 = ParsedQuestion(
            raw="q1",
            question_type="what",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["q1"],
            keywords=["q1"],
        )
        parsed_2 = ParsedQuestion(
            raw="q2",
            question_type="what",
            entities=[],
            noun_chunks=[],
            root_verb=None,
            sub_questions=["q2"],
            keywords=["q2"],
        )
        answer_obj = FinalAnswer(
            answer="ans", confidence=0.9, source="Wiki", url="http://e.com"
        )
        self.mock_nlp.process.side_effect = [parsed_1, parsed_2]
        self.mock_bm25.retrieve.return_value = []
        self.mock_vector.retrieve.return_value = []
        self.mock_ranker.rank.return_value = []
        self.mock_reader.extract.return_value = []
        self.mock_scorer.score.return_value = answer_obj

        # Initial passages are indexed on the first answer attempt.
        self.mock_fetch.return_value = [
            Passage(text="first passage", source="A", url="http://a.com")
        ]
        self.pipeline.answer("q1", verbose=False)

        # A different passage set should force a second indexing pass.
        self.mock_fetch.return_value = [
            Passage(text="different passage", source="B", url="http://b.com")
        ]
        self.pipeline.answer("q2", verbose=False)

        assert self.mock_bm25.index.call_count == 2
        assert self.mock_vector.index_passages.call_count == 2

    def test_vector_retrieval_toggle_off(self) -> None:
        self._setup_success_flow()
        self.pipeline = WatsonLite(
            config=self.base_config.with_feature("vector_retrieval", False)
        )

        self.pipeline.answer("test question", verbose=False)

        self.mock_vector.index_passages.assert_not_called()
        self.mock_vector.retrieve.assert_not_called()

    def test_query_expansion_toggle_off_uses_raw_question_once(self) -> None:
        self._setup_success_flow()
        self.pipeline = WatsonLite(
            config=self.base_config.with_feature("query_expansion", False)
        )

        self.pipeline.answer("test question", verbose=False)

        self.mock_fetch.assert_called_once_with("test question", top_k=5)

    def test_graph_enrichment_toggle_off(self) -> None:
        self._setup_success_flow()
        self.pipeline = WatsonLite(
            config=self.base_config.with_feature("graph_enrichment", False)
        )

        self.pipeline.answer("test question", verbose=False)

        self.mock_graph.enrich_all.assert_not_called()

    def test_cross_encoder_reranking_toggle_off(self) -> None:
        self._setup_success_flow()
        self.pipeline = WatsonLite(
            config=self.base_config.with_feature("cross_encoder_reranking", False)
        )

        self.pipeline.answer("test question", verbose=False)

        assert self.mock_ranker.rank.call_args is not None
        assert self.mock_ranker.rank.call_args.kwargs["use_cross_encoder"] is False

    def test_scoring_toggles_off(self) -> None:
        self._setup_success_flow()
        self.pipeline = WatsonLite(
            config=self.base_config.with_feature(
                "question_type_bonus", False
            ).with_feature("type_coercion", False)
        )

        self.pipeline.answer("test question", verbose=False)

        assert self.mock_scorer.score.call_args is not None
        assert (
            self.mock_scorer.score.call_args.kwargs["enable_question_type_bonus"]
            is False
        )
        assert self.mock_scorer.score.call_args.kwargs["enable_type_coercion"] is False

    def test_multi_hypothesis_adds_title_candidates(self) -> None:
        self._setup_success_flow()
        self.pipeline = WatsonLite(
            config=self.base_config.with_feature("multi_hypothesis", True)
        )

        self.pipeline.answer("test question", verbose=False)

        scored_candidates = self.mock_scorer.score.call_args.args[0]
        assert any(candidate.source == "title_match" for candidate in scored_candidates)
        assert any(candidate.span == "Eiffel Tower" for candidate in scored_candidates)

    def test_per_candidate_retrieval_disabled_skips_extra_query(self) -> None:
        passages = [Passage(text="Paris is in France.", source="Wiki", url="u")]
        ranked = [
            RankedPassage(
                passage=passages[0],
                rrf_score=1.0,
                cross_score=1.0,
                final_score=1.0,
                rank=1,
            )
        ]
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="Where is Paris?",
            question_type="where",
            entities=[],
            noun_chunks=["Paris"],
            root_verb="be",
            sub_questions=["Where is Paris?"],
            keywords=["paris"],
        )
        self.pipeline = WatsonLite(
            config=self.base_config.with_feature("query_expansion", False)
        )
        self.pipeline.dataset_query_engine.query = MagicMock(return_value=passages)
        self.mock_bm25.retrieve.return_value = passages
        self.mock_vector.retrieve.return_value = passages
        self.mock_ranker.rank.return_value = ranked
        self.mock_reader.extract.return_value = [
            AnswerCandidate(
                span="Paris",
                source="Wiki",
                url="u",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=1,
            )
        ]
        self.mock_scorer.score.return_value = FinalAnswer(
            answer="Paris", confidence=0.8, source="Wiki", url="u"
        )

        self.pipeline.answer("Where is Paris?", verbose=False)

        assert self.pipeline.dataset_query_engine.query.call_count == 1

    def test_iterative_retrieval_triggers_second_pass(self) -> None:
        initial_passages = [Passage(text="Paris is a city.", source="Wiki", url="u1")]
        refined_passages = [
            Passage(text="Paris is the capital of France.", source="Wiki", url="u2")
        ]
        initial_ranked = [
            RankedPassage(
                passage=initial_passages[0],
                rrf_score=1.0,
                cross_score=1.0,
                final_score=1.0,
                rank=1,
            )
        ]
        refined_ranked = [
            RankedPassage(
                passage=refined_passages[0],
                rrf_score=1.0,
                cross_score=1.0,
                final_score=1.0,
                rank=1,
            )
        ]
        self.mock_nlp.process.return_value = ParsedQuestion(
            raw="What is the capital of France?",
            question_type="what",
            entities=[],
            noun_chunks=["the capital of France"],
            root_verb="be",
            sub_questions=["What is the capital of France?"],
            keywords=["capital", "france"],
        )
        self.pipeline = WatsonLite(
            config=self.base_config.with_feature("iterative_retrieval", True)
            .with_feature("query_expansion", False)
            .with_feature("vector_retrieval", False)
        )
        self.pipeline.dataset_query_engine.query = MagicMock(
            side_effect=[initial_passages, refined_passages]
        )
        self.mock_bm25.retrieve.side_effect = [initial_passages, refined_passages]
        self.mock_ranker.rank.side_effect = [initial_ranked, refined_ranked]
        self.mock_reader.extract.side_effect = [
            (
                [
                    AnswerCandidate(
                        span="Paris",
                        source="Wiki",
                        url="u1",
                        passage="Paris is a city.",
                        extraction_score=0.4,
                        rank=1,
                    )
                ],
                0,
            ),
            (
                [
                    AnswerCandidate(
                        span="Paris",
                        source="Wiki",
                        url="u2",
                        passage="Paris is the capital of France.",
                        extraction_score=0.9,
                        rank=1,
                    )
                ],
                0,
            ),
        ]
        self.mock_scorer.score.side_effect = [
            FinalAnswer(answer="Paris", confidence=0.2, source="Wiki", url="u1"),
            FinalAnswer(answer="Paris", confidence=0.8, source="Wiki", url="u2"),
        ]

        result = self.pipeline.answer("What is the capital of France?", verbose=False)

        assert result.confidence == 0.8
        assert self.pipeline.dataset_query_engine.query.call_count == 2
        assert self.mock_scorer.score.call_count == 2
