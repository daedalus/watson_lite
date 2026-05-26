import pytest

from watson_lite.core.nlp import NLPProcessor


@pytest.fixture(scope="module")
def nlp():
    return NLPProcessor()


class TestQuestionClassification:
    def test_who(self, nlp) -> None:
        assert nlp.classify_question("Who designed the Eiffel Tower?") == "who"

    def test_what(self, nlp) -> None:
        assert nlp.classify_question("What is the capital of France?") == "what"

    def test_when(self, nlp) -> None:
        assert nlp.classify_question("When was it built?") == "when"

    def test_where(self, nlp) -> None:
        assert nlp.classify_question("Where is Paris?") == "where"

    def test_how(self, nlp) -> None:
        assert nlp.classify_question("How tall is it?") == "how"

    def test_why(self, nlp) -> None:
        assert nlp.classify_question("Why is the sky blue?") == "why"

    def test_unknown(self, nlp) -> None:
        assert nlp.classify_question("Really?") == "unknown"

    def test_empty(self, nlp) -> None:
        assert nlp.classify_question("") == "unknown"


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
        assert result.question_type == "who"
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
