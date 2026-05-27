from unittest.mock import MagicMock, patch

import pytest

from watson_lite.scoring.entailment import TextualEntailmentScorer, score_entailment


class TestTextualEntailmentScorer:
    def test_score_returns_max_entailment_probability(self) -> None:
        with patch("watson_lite.scoring.entailment.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.model.config.label2id = {
                "contradiction": 0,
                "entailment": 1,
                "neutral": 2,
            }
            mock_model.predict.return_value = [[0.0, 2.0, -1.0], [1.0, -1.0, 0.0]]
            mock_ce.return_value = mock_model
            scorer = TextualEntailmentScorer()

        score = scorer.score(
            "Who designed the Eiffel Tower?",
            "Gustave Eiffel",
            [
                "Gustave Eiffel designed the Eiffel Tower.",
                "The tower is in Paris.",
            ],
        )

        # Softmax([0, 2, -1])[entailment] ~= 0.844, and scorer takes max passage score.
        assert score == pytest.approx(0.844, abs=0.01)

    def test_score_uses_label_mapping_when_available(self) -> None:
        with patch("watson_lite.scoring.entailment.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.model.config.label2id = {
                "contradiction": 0,
                "neutral": 1,
                "entailment": 2,
            }
            mock_model.predict.return_value = [[0.0, 0.0, 3.0]]
            mock_ce.return_value = mock_model
            scorer = TextualEntailmentScorer()

        score = scorer.score("What is the capital of France?", "Paris", ["Paris is in France."])
        assert score > 0.8

    def test_score_returns_zero_for_empty_inputs(self) -> None:
        with patch("watson_lite.scoring.entailment.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.model.config.label2id = None
            mock_ce.return_value = mock_model
            scorer = TextualEntailmentScorer()
        assert scorer.score("", "Paris", ["Paris is in France."]) == 0.0
        assert scorer.score("Where is Paris?", "", ["Paris is in France."]) == 0.0
        assert scorer.score("Where is Paris?", "Paris", []) == 0.0


def test_score_entailment_returns_zero_when_import_missing() -> None:
    with (
        patch(
            "watson_lite.scoring.entailment.TextualEntailmentScorer",
            side_effect=ImportError,
        ),
        patch("watson_lite.scoring.entailment._ENTAILMENT_SCORER", None),
        patch("watson_lite.scoring.entailment._ENTAILMENT_UNAVAILABLE", False),
    ):
        assert score_entailment("Where is Paris?", "Paris", ["Paris is in France."]) == 0.0
