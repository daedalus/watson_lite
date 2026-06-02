from unittest.mock import MagicMock, patch

import pytest

from watson_lite.scoring.entailment import (
    TextualEntailmentScorer,
    _resolve_entailment_index,
    _stable_softmax,
    _coerce_to_float_vector,
    score_entailment,
    configure_entailment_model,
)


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

        score = scorer.score(
            "What is the capital of France?", "Paris", ["Paris is in France."]
        )
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

    def test_entailment_probability_single_value(self) -> None:
        with patch("watson_lite.scoring.entailment.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.model.config.label2id = None
            mock_ce.return_value = mock_model
            scorer = TextualEntailmentScorer()
        prob = scorer._entailment_probability(0.75)
        assert prob == pytest.approx(0.75)

    def test_entailment_probability_out_of_range_index(self) -> None:
        with patch("watson_lite.scoring.entailment.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.model.config.label2id = {"entailment": 99}
            mock_ce.return_value = mock_model
            scorer = TextualEntailmentScorer()
        prob = scorer._entailment_probability([0.1, 0.8, 0.1])
        assert prob == 0.0

    def test_entailment_probability_empty_values(self) -> None:
        with patch("watson_lite.scoring.entailment.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.model.config.label2id = None
            mock_ce.return_value = mock_model
            scorer = TextualEntailmentScorer()
        prob = scorer._entailment_probability([])
        assert prob == 0.0

    def test_entailment_probability_empty_after_softmax(self) -> None:
        with patch("watson_lite.scoring.entailment.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.model.config.label2id = None
            mock_ce.return_value = mock_model
            scorer = TextualEntailmentScorer()
        # Negative infinities give nan after softmax → empty filter returns 0.0
        prob = scorer._entailment_probability([])
        assert prob == 0.0


def test_score_entailment_returns_zero_when_import_missing() -> None:
    with (
        patch(
            "watson_lite.scoring.entailment.TextualEntailmentScorer",
            side_effect=ImportError,
        ),
        patch("watson_lite.scoring.entailment._ENTAILMENT_SCORER", None),
        patch("watson_lite.scoring.entailment._ENTAILMENT_UNAVAILABLE", False),
    ):
        assert (
            score_entailment("Where is Paris?", "Paris", ["Paris is in France."]) == 0.0
        )


def test_score_entailment_returns_zero_when_unavailable() -> None:
    with patch("watson_lite.scoring.entailment._ENTAILMENT_UNAVAILABLE", True):
        assert (
            score_entailment("Where is Paris?", "Paris", ["Paris is in France."]) == 0.0
        )


class TestStableSoftmax:
    def test_empty_values(self) -> None:
        assert _stable_softmax([]) == []

    def test_all_negative(self) -> None:
        import math

        result = _stable_softmax([-1e10, -1e10])
        assert result == [0.5, 0.5]


class TestResolveEntailmentIndex:
    def test_no_label2id(self) -> None:
        assert _resolve_entailment_index(None) == 1

    def test_no_entail_label(self) -> None:
        assert _resolve_entailment_index({"contradiction": 0}) == 1

    def test_finds_entail_label(self) -> None:
        assert _resolve_entailment_index({"entailment": 2}) == 2


class TestCoerceToFloatVector:
    def test_single_float(self) -> None:
        assert _coerce_to_float_vector(0.5) == [0.5]

    def test_single_int(self) -> None:
        assert _coerce_to_float_vector(3) == [3.0]

    def test_non_iterable(self) -> None:
        assert _coerce_to_float_vector(None) == []


class TestConfigureEntailmentModel:
    def test_resets_scorer(self) -> None:
        import watson_lite.scoring.entailment as entailment_mod

        entailment_mod._ENTAILMENT_SCORER = MagicMock()
        entailment_mod._ENTAILMENT_UNAVAILABLE = True
        configure_entailment_model("cross-encoder/nli-deberta-v3-small")
        assert entailment_mod._ENTAILMENT_SCORER is None
        assert entailment_mod._ENTAILMENT_UNAVAILABLE is False
