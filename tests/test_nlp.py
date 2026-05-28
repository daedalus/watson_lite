import pytest

from watson_lite.core.nlp import NLPProcessor


@pytest.fixture(scope="module")
def nlp():
    return NLPProcessor()


class TestQuestionClassification:
    def test_who(self, nlp) -> None:
        assert nlp.classify_question("Who designed the Eiffel Tower?") == "what"

    def test_what(self, nlp) -> None:
        assert nlp.classify_question("What is the capital of France?") == "what"

    def test_when(self, nlp) -> None:
        assert nlp.classify_question("When was it built?") == "what"

    def test_where(self, nlp) -> None:
        assert nlp.classify_question("Where is Paris?") == "what"

    def test_how(self, nlp) -> None:
        assert nlp.classify_question("How tall is it?") == "what"

    def test_why(self, nlp) -> None:
        assert nlp.classify_question("Why is the sky blue?") == "what"

    def test_spanish_por_que(self) -> None:
        es = NLPProcessor(language="es")
        assert es.classify_question("¿Por qué cayó el Imperio Romano?") == "what"

    def test_unknown(self, nlp) -> None:
        assert nlp.classify_question("Really?") == "unknown"

    def test_empty(self, nlp) -> None:
        assert nlp.classify_question("") == "unknown"

    def test_expletive_pronoun(self, nlp) -> None:
        assert nlp.classify_question("There is a book") == "unknown"

    def test_adp_non_interrogative(self, nlp) -> None:
        assert nlp.classify_question("Go to school") == "unknown"


class TestExtractEntities:
    def test_filters_entity_with_verb(self, nlp) -> None:
        import spacy
        doc = nlp.nlp("Who built the Eiffel Tower?")
        ents = nlp.extract_entities(doc)
        for ent in ents:
            assert "built" not in ent["text"].lower()

    def test_spanish_entity_without_verb(self) -> None:
        import spacy
        try:
            es = spacy.load("es_core_news_sm")
        except OSError:
            pytest.skip("es_core_news_sm not installed")
        nlp_es = NLPProcessor(language="es")
        text = "¿Qué ciudad es la capital de Francia?"
        from watson_lite.core.nlp import _ner_input
        normalized = _ner_input(text, es, language="es")
        doc = es(normalized)
        ents = nlp_es.extract_entities(doc)
        assert len(ents) > 0
        for ent in ents:
            assert "es" not in ent["text"].lower()


class TestNLPProcessorMisc:
    def test_int_list_valid(self, nlp) -> None:
        result = NLPProcessor._int_list([1, 2, 3])
        assert result == [1, 2, 3]

    def test_int_list_mixed(self, nlp) -> None:
        result = NLPProcessor._int_list([1, "a", 3])
        assert result == [1, 3]

    def test_int_list_empty(self, nlp) -> None:
        result = NLPProcessor._int_list([])
        assert result is None

    def test_int_list_not_list(self, nlp) -> None:
        result = NLPProcessor._int_list("hello")
        assert result is None

    def test_int_list_no_ints(self, nlp) -> None:
        result = NLPProcessor._int_list(["a", "b"])
        assert result is None


class TestDecompose:
    def test_single(self, nlp) -> None:
        result = nlp.decompose_question("Who built the Eiffel Tower?")
        assert result == ["Who built the Eiffel Tower?"]

    def test_and(self, nlp) -> None:
        result = nlp.decompose_question("Who built it and when was it built?")
        assert len(result) == 2

    def test_short_conjunct(self, nlp) -> None:
        result = nlp.decompose_question("A and B")
        assert result == ["A and B"]


class TestProcess:
    def test_basic_process(self, nlp) -> None:
        result = nlp.process("Who designed the Eiffel Tower?")
        assert result.question_type == "what"
        assert len(result.entities) > 0
        assert len(result.sub_questions) == 1

    def test_case_insensitive_entity_detection(self, nlp) -> None:
        upper = nlp.process("Who was the Norse leader?")
        lower = nlp.process("who was the norse leader?")
        assert len(upper.entities) > 0
        assert len(upper.entities) == len(lower.entities)
        for ue, le in zip(upper.entities, lower.entities):
            assert ue["label"] == le["label"]
            assert ue["text"].lower() == le["text"].lower()
