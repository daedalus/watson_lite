"""Tests for NLP utility functions that extract lexical answer types."""

import pytest

from watson_lite.core.nlp import _extract_lat


@pytest.fixture(scope="module")
def nlp():
    import spacy

    return spacy.load("en_core_web_sm")


class TestExtractLat:
    def test_who_returns_person(self, nlp) -> None:
        doc = nlp("Who invented the telephone?")
        lat, qids = _extract_lat(doc)
        assert lat == "person"
        assert "Q5" in qids

    def test_where_extracts_proper_noun(self, nlp) -> None:
        doc = nlp("Where is the Eiffel Tower?")
        lat, qids = _extract_lat(doc)
        assert lat == "eiffel"
        assert qids == []

    def test_when_extracts_entity(self, nlp) -> None:
        doc = nlp("When was Shakespeare born?")
        lat, qids = _extract_lat(doc)
        assert lat == "shakespeare"
        assert qids == []

    def test_why_extracts_entity(self, nlp) -> None:
        doc = nlp("Why did Rome fall?")
        lat, qids = _extract_lat(doc)
        assert lat == "rome"
        assert qids == []

    def test_what_known_noun_returns_lat(self, nlp) -> None:
        doc = nlp("What country is France?")
        lat, qids = _extract_lat(doc)
        assert lat == "country"
        assert "Q6256" in qids

    def test_what_skip_word_finds_head(self, nlp) -> None:
        doc = nlp("What is the capital of France?")
        lat, qids = _extract_lat(doc)
        assert lat == "capital"
        assert qids == []

    def test_what_unknown_noun_returns_headword(self, nlp) -> None:
        doc = nlp("What thing happened?")
        lat, qids = _extract_lat(doc)
        assert lat == "thing"
        assert qids == []

    def test_which_known_noun(self, nlp) -> None:
        doc = nlp("Which planet is the largest?")
        lat, qids = _extract_lat(doc)
        assert lat == "planet"
        assert qids

    def test_unknown_question_type(self, nlp) -> None:
        doc = nlp("How tall is the tower?")
        lat, qids = _extract_lat(doc)
        assert lat == "tower"
        assert qids == []


class TestExtractLatSpanish:
    def test_quien_returns_person(self, nlp) -> None:
        import spacy

        try:
            es = spacy.load("es_core_news_sm")
        except OSError:
            pytest.skip("es_core_news_sm not installed")
        doc = es("¿Quién diseñó la Torre Eiffel?")
        lat, qids = _extract_lat(doc)
        assert lat == "person"
        assert "Q5" in qids

    def test_que_finds_noun(self, nlp) -> None:
        import spacy

        try:
            es = spacy.load("es_core_news_sm")
        except OSError:
            pytest.skip("es_core_news_sm not installed")
        doc = es("¿Qué ciudad es la capital de Francia?")
        lat, qids = _extract_lat(doc)
        assert lat == "ciudad"
        assert qids == []

    def test_donde(self, nlp) -> None:
        import spacy

        try:
            es = spacy.load("es_core_news_sm")
        except OSError:
            pytest.skip("es_core_news_sm not installed")
        doc = es("¿Dónde está París?")
        lat, qids = _extract_lat(doc)
        assert lat == "parís"
        assert qids == []
