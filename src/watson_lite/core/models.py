from dataclasses import dataclass, field


@dataclass
class Passage:
    text: str
    source: str
    url: str
    score: float = 0.0
    rank: int = 0


@dataclass
class RankedPassage:
    passage: Passage
    rrf_score: float = 0.0
    cross_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0


@dataclass
class AnswerCandidate:
    span: str
    source: str
    url: str
    passage: str
    extraction_score: float
    rank: int
    graph_corroborated: bool = False


@dataclass
class FinalAnswer:
    answer: str
    confidence: float
    source: str
    url: str
    supporting_passages: list[str] = field(default_factory=list)
    graph_facts: list[str] = field(default_factory=list)
    confidence_breakdown: dict[str, float | str] = field(default_factory=dict)


@dataclass
class EntityFact:
    entity: str
    property_label: str
    value: str
    value_type: str = "literal"


@dataclass
class GraphResult:
    entity_name: str
    wikidata_id: str | None
    facts: list[EntityFact] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)


@dataclass
class ParsedQuestion:
    raw: str
    question_type: str
    entities: list[dict[str, str | int]]
    noun_chunks: list[str]
    root_verb: str | None
    sub_questions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    lat: str | None = None
    lat_qids: list[str] = field(default_factory=list)
