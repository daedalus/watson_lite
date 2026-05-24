import pytest

from watson_lite.core.extractor import ConfidenceScorer, _question_type_bonus
from watson_lite.core.models import AnswerCandidate, EntityFact, GraphResult


class TestConfidenceScorer:
    def setup_method(self) -> None:
        self.scorer = ConfidenceScorer()

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
                passage="",
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
                passage="",
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
            AnswerCandidate(
                span="London",
                source="c",
                url="",
                passage="",
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
                passage="",
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
                passage="",
                extraction_score=0.9,
                rank=1,
            ),
        ]
        result = self.scorer.score(candidates, [], "where")
        assert "question_type_bonus" in result.confidence_breakdown


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
