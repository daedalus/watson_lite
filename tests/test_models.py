from watson_lite.core.models import (
    AnswerCandidate,
    EntityFact,
    FinalAnswer,
    GraphResult,
    ParsedQuestion,
    Passage,
    RankedPassage,
)


class TestPassage:
    def test_default_score(self) -> None:
        p = Passage(text="t", source="s", url="u")
        assert p.score == 0.0
        assert p.rank == 0

    def test_full_construction(self) -> None:
        p = Passage(text="hello", source="src", url="http://e.x", score=0.5, rank=2)
        assert p.text == "hello"
        assert p.source == "src"
        assert p.url == "http://e.x"
        assert p.score == 0.5
        assert p.rank == 2


class TestRankedPassage:
    def test_defaults(self, sample_passage) -> None:
        rp = RankedPassage(passage=sample_passage)
        assert rp.rrf_score == 0.0
        assert rp.cross_score == 0.0
        assert rp.final_score == 0.0
        assert rp.rank == 0


class TestAnswerCandidate:
    def test_defaults(self) -> None:
        ac = AnswerCandidate(
            span="x", source="s", url="u", passage="p", extraction_score=0.5, rank=1
        )
        assert ac.graph_corroborated is False
        assert ac.doc_frequency == 1


class TestFinalAnswer:
    def test_default_lists(self) -> None:
        fa = FinalAnswer(answer="a", confidence=0.5, source="s", url="u")
        assert fa.supporting_passages == []
        assert fa.graph_facts == []
        assert fa.confidence_breakdown == {}
        assert fa.evidence_chain == []


class TestEntityFact:
    def test_default_type(self) -> None:
        ef = EntityFact(entity="Q1", property_label="label", value="val")
        assert ef.value_type == "literal"


class TestGraphResult:
    def test_defaults(self) -> None:
        gr = GraphResult(entity_name="Eiffel Tower", wikidata_id="Q243")
        assert gr.facts == []
        assert gr.related_entities == []


class TestParsedQuestion:
    def test_defaults(self) -> None:
        pq = ParsedQuestion(
            raw="q", question_type="who", entities=[], noun_chunks=[], root_verb=None
        )
        assert pq.sub_questions == []
        assert pq.keywords == []
        assert pq.srl_frames == []
        assert pq.coref_clusters == []
