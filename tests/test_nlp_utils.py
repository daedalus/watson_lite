"""Tests for NLP utility functions that extract lexical answer types."""

import pytest

from watson_lite.core.nlp import (
    _classify_question_word,
    _detect_question_word,
    _extract_lat,
)


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


class TestDetectQuestionWord:
    def test_english_who(self, nlp) -> None:
        doc = nlp("Who built the Eiffel Tower?")
        assert _detect_question_word(doc) == "who"

    def test_english_what(self, nlp) -> None:
        doc = nlp("What is the capital of France?")
        assert _detect_question_word(doc) == "what"

    def test_english_when(self, nlp) -> None:
        doc = nlp("When was it built?")
        assert _detect_question_word(doc) == "when"

    def test_english_where(self, nlp) -> None:
        doc = nlp("Where is Paris?")
        assert _detect_question_word(doc) == "where"

    def test_english_why(self, nlp) -> None:
        doc = nlp("Why is the sky blue?")
        assert _detect_question_word(doc) == "why"

    def test_english_how(self, nlp) -> None:
        doc = nlp("How tall is it?")
        assert _detect_question_word(doc) == "how"

    def test_english_which(self, nlp) -> None:
        doc = nlp("Which planet is the largest?")
        assert _detect_question_word(doc) == "which"

    def test_english_unknown(self, nlp) -> None:
        doc = nlp("Really?")
        assert _detect_question_word(doc) is None

    def test_spanish_quien(self, nlp) -> None:
        import spacy

        try:
            es = spacy.load("es_core_news_sm")
        except OSError:
            pytest.skip("es_core_news_sm not installed")
        doc = es("¿Quién diseñó la Torre Eiffel?")
        assert _detect_question_word(doc) == "quién"

    def test_spanish_por_que(self, nlp) -> None:
        import spacy

        try:
            es = spacy.load("es_core_news_sm")
        except OSError:
            pytest.skip("es_core_news_sm not installed")
        doc = es("¿Por qué cayó el Imperio Romano?")
        assert _detect_question_word(doc) == "por qué"


class TestClassifyQuestionWord:
    def test_person_who(self, nlp) -> None:
        doc = nlp("Who built the Eiffel Tower?")
        assert _classify_question_word(doc, "who") == "person"

    def test_person_whom(self, nlp) -> None:
        doc = nlp("Whom did you meet?")
        assert _classify_question_word(doc, "whom") == "person"

    def test_person_whose(self, nlp) -> None:
        doc = nlp("Whose book is this?")
        assert _classify_question_word(doc, "whose") == "person"

    def test_time_when(self, nlp) -> None:
        doc = nlp("When was the Eiffel Tower built?")
        assert _classify_question_word(doc, "when") == "time"

    def test_time_cuando(self, nlp) -> None:
        import spacy

        try:
            es = spacy.load("es_core_news_sm")
        except OSError:
            pytest.skip("es_core_news_sm not installed")
        doc = es("¿Cuándo se construyó la Torre Eiffel?")
        assert _classify_question_word(doc, "cuándo") == "time"

    def test_person_quien(self, nlp) -> None:
        import spacy

        try:
            es = spacy.load("es_core_news_sm")
        except OSError:
            pytest.skip("es_core_news_sm not installed")
        doc = es("¿Quién construyó la Torre Eiffel?")
        assert _classify_question_word(doc, "quién") == "person"

    def test_none_returns_none(self, nlp) -> None:
        doc = nlp("Hello world")
        assert _classify_question_word(doc, None) is None

    def test_no_match_returns_none(self, nlp) -> None:
        doc = nlp("Hello world")
        assert _classify_question_word(doc, "who") is None

    def test_what_returns_person(self, nlp) -> None:
        doc = nlp("What is the capital of France?")
        assert _classify_question_word(doc, "what") == "person"

    def test_where_returns_time(self, nlp) -> None:
        doc = nlp("Where is Paris?")
        assert _classify_question_word(doc, "where") == "time"

    def test_det_returns_person(self, nlp) -> None:
        doc = nlp("Which planet?")
        assert _classify_question_word(doc, "which") == "person"

    def test_unknown_pos_returns_none(self, nlp) -> None:
        doc = nlp("Hello world")
        assert _classify_question_word(doc, "hello") is None


class TestDetectQuestionWordEdgeCases:
    def test_non_interrogative_pronoun(self, nlp) -> None:
        doc = nlp("I am here")
        assert _detect_question_word(doc) is None

    def test_adp_non_interrogative(self, nlp) -> None:
        doc = nlp("Give to me")
        assert _detect_question_word(doc) is None

    def test_empty_doc(self, nlp) -> None:
        doc = nlp("")
        assert _detect_question_word(doc) is None
