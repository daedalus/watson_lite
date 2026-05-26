"""Tests for pure-Python NLP utility functions that do not require spaCy models."""

from watson_lite.core.nlp import _extract_lat


class TestExtractLat:
    def test_who_returns_person(self) -> None:
        lat, qids = _extract_lat("Who invented the telephone?", "who")
        assert lat == "person"
        assert "Q5" in qids

    def test_where_returns_location(self) -> None:
        lat, qids = _extract_lat("Where is the Eiffel Tower?", "where")
        assert lat == "location"
        assert qids  # should have city QIDs

    def test_when_returns_none(self) -> None:
        lat, qids = _extract_lat("When was Shakespeare born?", "when")
        assert lat is None
        assert qids == []

    def test_why_returns_none(self) -> None:
        lat, qids = _extract_lat("Why did Rome fall?", "why")
        assert lat is None
        assert qids == []

    def test_what_known_noun_returns_lat(self) -> None:
        lat, qids = _extract_lat("What country is France?", "what")
        assert lat == "country"
        assert qids  # should contain Q6256

    def test_what_skip_word_returns_none(self) -> None:
        # "is" is in skip_words so LAT lookup should fail
        lat, qids = _extract_lat("What is the capital of France?", "what")
        assert lat is None
        assert qids == []

    def test_what_unknown_noun_not_in_map(self) -> None:
        # "thing" is not in LAT_QID_MAP
        lat, qids = _extract_lat("What thing happened?", "what")
        assert lat is None
        assert qids == []

    def test_which_known_noun(self) -> None:
        lat, qids = _extract_lat("Which planet is the largest?", "what")
        assert lat == "planet"
        assert qids  # should contain Q634

    def test_unknown_question_type(self) -> None:
        lat, qids = _extract_lat("How tall is the tower?", "how")
        assert lat is None
        assert qids == []
