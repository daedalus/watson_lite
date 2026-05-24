import pytest

from watson_lite.core.models import (
    AnswerCandidate,
    EntityFact,
    GraphResult,
    Passage,
    RankedPassage,
)


@pytest.fixture
def sample_passage():
    return Passage(
        text="The Eiffel Tower was designed by Gustave Eiffel and built between 1887 and 1889.",
        source="Eiffel Tower",
        url="https://en.wikipedia.org/wiki/Eiffel_Tower",
        score=0.9,
        rank=1,
    )


@pytest.fixture
def sample_ranked(sample_passage):
    return RankedPassage(
        passage=sample_passage, rrf_score=0.8, cross_score=0.9, final_score=0.9, rank=1
    )


@pytest.fixture
def sample_candidates():
    return [
        AnswerCandidate(
            span="Gustave Eiffel",
            source="Eiffel Tower",
            url="",
            passage="",
            extraction_score=0.97,
            rank=1,
        ),
        AnswerCandidate(
            span="Gustave Eiffel",
            source="Wikipedia",
            url="",
            passage="",
            extraction_score=0.85,
            rank=2,
        ),
        AnswerCandidate(
            span="Eiffel",
            source="Eiffel Tower",
            url="",
            passage="",
            extraction_score=0.70,
            rank=3,
        ),
    ]


@pytest.fixture
def sample_graph_results():
    return [
        GraphResult(
            entity_name="Eiffel Tower",
            wikidata_id="Q243",
            facts=[
                EntityFact(
                    entity="Q243", property_label="architect", value="Gustave Eiffel"
                )
            ],
        )
    ]
