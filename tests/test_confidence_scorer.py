from unittest.mock import patch

import pytest

from watson_lite.core.extractor import ConfidenceScorer, _question_type_bonus
from watson_lite.core.models import AnswerCandidate, EntityFact, GraphResult


class TestConfidenceScorer:
    def setup_method(self) -> None:
        self.scorer = ConfidenceScorer()
        self.merge_resolve_patcher = patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid"
        )
        self.extractor_resolve_patcher = patch(
            "watson_lite.core.extractor.resolve_span_to_qid"
        )
        self.mock_merge_resolve = self.merge_resolve_patcher.start()
        self.mock_extractor_resolve = self.extractor_resolve_patcher.start()
        self.mock_merge_resolve.return_value = None
        self.mock_extractor_resolve.return_value = None

    def teardown_method(self) -> None:
        self.merge_resolve_patcher.stop()
        self.extractor_resolve_patcher.stop()

    def test_no_candidates(self) -> None:
        result = self.scorer.score([], [], "who")
        assert result.answer == "No answer found"
        assert result.confidence == 0.0

    def test_single_candidate(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="src",
                url="",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=1,
            )
        ]
        result = self.scorer.score(candidates, [], "where")
        assert result.answer == "Paris"
        assert result.confidence > 0.0

    def test_graph_corroboration(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="src",
                url="",
                passage="Gustave Eiffel designed the tower.",
                extraction_score=0.9,
                rank=1,
            )
        ]
        graph = [
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
        result = self.scorer.score(candidates, graph, "who")
        assert "architect: Gustave Eiffel" in result.graph_facts
        assert result.confidence_breakdown["graph_corroboration"] == 0.2

    def test_span_agreement(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="a",
                url="",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=1,
            ),
            AnswerCandidate(
                span="Paris",
                source="b",
                url="",
                passage="Paris is also called the City of Light.",
                extraction_score=0.8,
                rank=2,
            ),
            AnswerCandidate(
                span="London",
                source="c",
                url="",
                passage="London is in the UK.",
                extraction_score=0.7,
                rank=3,
            ),
        ]
        result = self.scorer.score(candidates, [], "where")
        assert result.answer == "Paris"
        assert result.confidence_breakdown["span_agreement"] == pytest.approx(
            0.667, abs=0.01
        )

    def test_rank_signal_penalty(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="a",
                url="",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=6,
            ),
        ]
        result = self.scorer.score(candidates, [], "where")
        assert result.confidence_breakdown["passage_rank_signal"] == 0.5

    def test_confidence_breakdown_includes_question_type_bonus(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="a",
                url="",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        result = self.scorer.score(candidates, [], "where")
        assert "question_type_bonus" in result.confidence_breakdown

    def test_question_type_bonus_toggle_off(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="a",
                url="",
                passage="Gustave Eiffel designed the tower.",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        result = self.scorer.score(
            candidates,
            [],
            "who",
            enable_question_type_bonus=False,
        )
        assert result.confidence_breakdown["question_type_bonus"] == 0.0

    def test_type_coercion_toggle_off(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="a",
                url="",
                passage="Gustave Eiffel designed the tower.",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        with patch("watson_lite.core.extractor.score_type_coercion") as mock_tc:
            result = self.scorer.score(
                candidates,
                [],
                "who",
                lat_qids=["Q5"],
                enable_type_coercion=False,
            )
            mock_tc.assert_not_called()
            assert result.confidence_breakdown["type_coercion"] == 0.0

    def test_entailment_toggle_off(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="a",
                url="",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        with patch("watson_lite.core.extractor.score_entailment") as mock_ent:
            result = self.scorer.score(
                candidates,
                [],
                "where",
                question="Where is Paris?",
                enable_entailment=False,
            )
            mock_ent.assert_not_called()
            assert result.confidence_breakdown["textual_entailment"] == 0.0

    def test_entailment_signal_affects_breakdown(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="a",
                url="",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=1,
            )
        ]

        without_signal = self.scorer.score(
            candidates, [], "where", question="Where is Paris?", enable_entailment=False
        )
        with patch("watson_lite.core.extractor.score_entailment", return_value=1.0):
            with_signal = self.scorer.score(
                candidates, [], "where", question="Where is Paris?"
            )

        assert with_signal.confidence > without_signal.confidence
        assert with_signal.confidence_breakdown["textual_entailment"] == 1.0

    def test_doc_frequency_signal_increases_confidence(self) -> None:
        low = self.scorer.score(
            [
                AnswerCandidate(
                    span="Paris",
                    source="a",
                    url="",
                    passage="Paris is the capital of France.",
                    extraction_score=0.9,
                    rank=1,
                    doc_frequency=1,
                ),
                AnswerCandidate(
                    span="London",
                    source="b",
                    url="",
                    passage="London is the capital of the United Kingdom.",
                    extraction_score=0.4,
                    rank=2,
                    doc_frequency=3,
                ),
            ],
            [],
            "where",
        )

        high = self.scorer.score(
            [
                AnswerCandidate(
                    span="Paris",
                    source="a",
                    url="",
                    passage="Paris is the capital of France.",
                    extraction_score=0.9,
                    rank=1,
                    doc_frequency=4,
                ),
                AnswerCandidate(
                    span="London",
                    source="b",
                    url="",
                    passage="London is the capital of the United Kingdom.",
                    extraction_score=0.4,
                    rank=2,
                    doc_frequency=1,
                ),
            ],
            [],
            "where",
        )

        assert high.confidence > low.confidence
        # With doc_frequency=1 and a max competitor frequency of 3, the
        # normalized signal is 1 / 3 ≈ 0.333.
        assert low.confidence_breakdown["frequency_signal"] == pytest.approx(
            0.333, abs=0.01
        )
        assert high.confidence_breakdown["frequency_signal"] == 1.0

    def test_evidence_chain_is_populated(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Gustave Eiffel",
                source="a",
                url="",
                passage="The Eiffel Tower is in Paris. Gustave Eiffel designed it.",
                extraction_score=0.9,
                rank=1,
            )
        ]
        graph = [
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

        result = self.scorer.score(candidates, graph, "who")

        assert result.evidence_chain
        assert result.evidence_chain[0].span == "Gustave Eiffel"
        assert any(item.graph_property == "architect" for item in result.evidence_chain)

    def test_type_mismatch_gates_qt_bonus_and_penalizes_confidence(self) -> None:
        with (
            patch(
                "watson_lite.core.extractor.resolve_span_to_qid",
                return_value="Q207440",
            ),
            patch(
                "watson_lite.core.extractor.score_type_coercion",
                return_value=0.0,
            ),
        ):
            candidates = [
                AnswerCandidate(
                    span="Eli Lilly",
                    source="src",
                    url="",
                    passage="Eli Lilly manufactured penicillin.",
                    extraction_score=0.976,
                    rank=1,
                ),
            ]
            result = self.scorer.score(candidates, [], "who", lat_qids=["Q5"])
        assert result.confidence_breakdown["question_type_bonus"] == 0.0
        assert result.confidence_breakdown["type_coercion"] == 0.0
        assert result.confidence < 0.3

    def test_threshold_abstains_when_confidence_below(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="src",
                url="",
                passage="Paris is in France.",
                extraction_score=0.05,
                rank=10,
            )
        ]
        scorer = ConfidenceScorer(confidence_threshold=0.9)
        result = scorer.score(candidates, [], "where")
        assert result.answer == "I don't know"
        assert result.confidence_breakdown["reason"] == "below_threshold"
        assert result.confidence_breakdown["threshold"] == 0.9

    def test_threshold_passes_when_confidence_above(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="src",
                url="",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=1,
            )
        ]
        scorer = ConfidenceScorer(confidence_threshold=0.0)
        result = scorer.score(candidates, [], "where")
        assert result.answer == "Paris"

    def test_no_threshold_always_returns_answer(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="src",
                url="",
                passage="Paris is in France.",
                extraction_score=0.01,
                rank=10,
            )
        ]
        scorer = ConfidenceScorer(confidence_threshold=None)
        result = scorer.score(candidates, [], "where")
        assert result.answer == "Paris"

    def test_threshold_abstain_preserves_computed_confidence(self) -> None:
        candidates = [
            AnswerCandidate(
                span="Paris",
                source="src",
                url="",
                passage="Paris is in France.",
                extraction_score=0.05,
                rank=10,
            )
        ]
        scorer_no_gate = ConfidenceScorer(confidence_threshold=None)
        scorer_with_gate = ConfidenceScorer(confidence_threshold=0.9)
        result_raw = scorer_no_gate.score(candidates, [], "where")
        result_gated = scorer_with_gate.score(candidates, [], "where")
        # The stored confidence value should equal the raw computed confidence
        assert result_gated.confidence == pytest.approx(result_raw.confidence)


        candidates = [
            AnswerCandidate(
                span="Paris",
                source="a",
                url="",
                passage="Paris is in France.",
                extraction_score=0.9,
                rank=1,
            )
        ]

        without_signal = self.scorer.score(candidates, [], "where")
        with_signal = self.scorer.score(
            candidates,
            [],
            "where",
            bidirectional_signal=1.0,
        )

        assert with_signal.confidence > without_signal.confidence
        assert with_signal.confidence_breakdown["bidirectional_signal"] == 1.0


class TestQuestionTypeBonus:
    def test_who_multi_word_capitalized(self) -> None:
        assert _question_type_bonus("Gustave Eiffel", "who") == 0.1

    def test_who_single_word(self) -> None:
        assert _question_type_bonus("Paris", "who") == 0.0

    def test_when_year(self) -> None:
        assert _question_type_bonus("1889", "when") == 0.1

    def test_when_month_year(self) -> None:
        assert _question_type_bonus("March 1889", "when") == 0.1

    def test_when_no_date(self) -> None:
        assert _question_type_bonus("Gustave Eiffel", "when") == 0.0

    def test_where_returns_zero(self) -> None:
        assert _question_type_bonus("Paris", "where") == 0.0

    def test_unknown_type(self) -> None:
        assert _question_type_bonus("anything", "unknown") == 0.0
