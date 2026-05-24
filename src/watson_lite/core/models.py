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
class AnswerDiagnostics:
    total_latency_s: float = 0.0
    stage_latencies_s: dict[str, float] = field(default_factory=dict)
    passages_fetched: int = 0
    passages_reranked: int = 0
    passages_extracted: int = 0
    retrieval_empty: bool = False
    extraction_errors: int = 0
    fallback_answer: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hits_by_namespace: dict[str, int] = field(default_factory=dict)
    cache_misses_by_namespace: dict[str, int] = field(default_factory=dict)
    top_retrieved_passages: list[str] = field(default_factory=list)


@dataclass
class FinalAnswer:
    answer: str
    confidence: float
    source: str
    url: str
    supporting_passages: list[str] = field(default_factory=list)
    graph_facts: list[str] = field(default_factory=list)
    confidence_breakdown: dict[str, float | str] = field(default_factory=dict)
    diagnostics: AnswerDiagnostics | None = None


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
