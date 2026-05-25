from unittest.mock import MagicMock

from watson_lite.core.models import Passage
from watson_lite.scoring.double_check import bidirectional_score


def test_bidirectional_score_empty_span() -> None:
    engine = MagicMock()
    assert bidirectional_score("", "Who built the Eiffel Tower?", engine) == 0.0


def test_bidirectional_score_no_keyword_overlap() -> None:
    engine = MagicMock()
    engine.query.return_value = [
        Passage(text="A mountain range in Asia.", source="wiki", url="u"),
        Passage(text="An ocean current.", source="wiki", url="u2"),
        Passage(text="A composer biography.", source="wiki", url="u3"),
    ]

    score = bidirectional_score("Paris", "Who built the Eiffel Tower?", engine)

    assert score == 0.0


def test_bidirectional_score_all_passages_match_keywords() -> None:
    engine = MagicMock()
    engine.query.return_value = [
        Passage(
            text="Paris is tied to the Eiffel Tower history.", source="wiki", url="u"
        ),
        Passage(text="The tower in Paris remains iconic.", source="wiki", url="u2"),
        Passage(
            text="Visitors ask who built the tower in Paris.", source="wiki", url="u3"
        ),
    ]

    score = bidirectional_score("Paris", "Who built the Eiffel Tower?", engine)

    assert score == 1.0


def test_bidirectional_score_handles_empty_keywords() -> None:
    engine = MagicMock()
    engine.query.return_value = [Passage(text="Anything", source="wiki", url="u")]

    assert bidirectional_score("Paris", "is the and or", engine) == 0.0


def test_bidirectional_score_empty_passages_returns_zero() -> None:
    """When the dataset engine returns no passages, score should be 0.0."""
    engine = MagicMock()
    engine.query.return_value = []

    score = bidirectional_score("Eiffel", "Who built the Eiffel Tower?", engine)

    assert score == 0.0
