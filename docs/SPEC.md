# SPEC.md — watson-lite

## Purpose

WatsonLite is an extractive question-answering system inspired by IBM Watson.
It answers factual questions using Wikipedia as its knowledge base — with no
LLM, no trained weights of its own, and no paid APIs. All components are
off-the-shelf pretrained models running CPU-only inference.

## Scope

### What IS in scope

- Answer single factual questions (who, what, when, where, how, why)
- Retrieve relevant passages from Wikipedia via REST API
- Rerank passages using BM25 + dense vector + cross-encoder fusion
- Enrich answers with Wikidata structured facts
- Confidence scoring using multiple signals (extraction score, span agreement, graph corroboration, rank signal)
- SQLite3 cache for Wikipedia and Wikidata responses
- Interactive CLI and single-shot CLI modes
- Extractive (span-based) answers — no text generation

### What is NOT in scope

- Generative / LLM-based answers
- Multi-turn conversational memory
- Document upload or custom corpus management
- Training or fine-tuning of any model
- Streaming answers
- Non-English questions (English only)

## Public API / Interface

### `watson_lite.pipeline.WatsonLite`

```python
class WatsonLite:
    def __init__(self) -> None: ...
    def answer(self, question: str, verbose: bool = True) -> FinalAnswer: ...
```

- `__init__`: Loads all subcomponents (NLP, BM25, vector, graph, ranker, reader, scorer).
  Raises `OSError` if a pretrained model cannot be loaded.
- `answer`: Runs the full 6-stage pipeline. Returns a `FinalAnswer`.
  Raises `ValueError` if `question` is empty.

### `watson_lite.core.nlp.NLPProcessor`

```python
class NLPProcessor:
    def __init__(self, model: str = "en_core_web_sm") -> None: ...
    def process(self, question: str) -> ParsedQuestion: ...
    def classify_question(self, text: str) -> str: ...
    def decompose_question(self, text: str) -> list[str]: ...
```

### `watson_lite.retrieval.bm25_retriever.BM25Retriever`

```python
class BM25Retriever:
    def __init__(self) -> None: ...
    def index(self, passages: list[Passage]) -> None: ...
    def retrieve(self, query: str, top_k: int = 10) -> list[Passage]: ...
    def fetch_and_retrieve(self, query: str, top_k: int = 10) -> list[Passage]: ...
```

### Module-level function

```python
def fetch_wikipedia_passages(query: str, top_k: int = 5) -> list[Passage]: ...
```

### `watson_lite.retrieval.vector_retriever.VectorRetriever`

```python
class VectorRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None: ...
    def index_passages(self, passages: list[Passage]) -> None: ...
    def retrieve(self, query: str, top_k: int = 10) -> list[Passage]: ...
```

### `watson_lite.graph.wikidata.WikidataGraph`

```python
class WikidataGraph:
    def __init__(self) -> None: ...
    def enrich(self, entity_name: str) -> GraphResult: ...
    def enrich_all(self, entity_names: list[str]) -> list[GraphResult]: ...
```

### `watson_lite.ranking.ranker.Ranker`

```python
class Ranker:
    def __init__(self) -> None: ...
    def rank(
        self,
        query: str,
        bm25_results: list[Passage],
        vector_results: list[Passage],
        top_k: int = 10,
    ) -> list[RankedPassage]: ...
```

### `watson_lite.core.extractor.ExtractiveReader`

```python
class ExtractiveReader:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2") -> None: ...
    def extract(self, question: str, passages: list[RankedPassage], top_k: int = 5) -> list[AnswerCandidate]: ...
```

### `watson_lite.core.extractor.ConfidenceScorer`

```python
class ConfidenceScorer:
    def score(
        self,
        candidates: list[AnswerCandidate],
        graph_results: list[GraphResult],
        question_type: str,
    ) -> FinalAnswer: ...
```

### `watson_lite.core.cache.Cache`

```python
class Cache:
    def __init__(self, db_path: str = "watson_lite_cache.sqlite3") -> None: ...
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...
```

### CLI

```bash
# Single question
python -m watson_lite "Who designed the Eiffel Tower?"

# Interactive mode
python -m watson_lite
```

## Data Formats

### Core dataclasses (`watson_lite.core.models`)

```python
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
    supporting_passages: list[str]
    graph_facts: list[str]
    confidence_breakdown: dict
    diagnostics: AnswerDiagnostics | None

@dataclass
class AnswerDiagnostics:
    total_latency_s: float
    stage_latencies_s: dict[str, float]
    passages_fetched: int
    passages_reranked: int
    passages_extracted: int
    retrieval_empty: bool
    extraction_errors: int
    fallback_answer: bool
    cache_hits: int
    cache_misses: int
    cache_hits_by_namespace: dict[str, int]
    cache_misses_by_namespace: dict[str, int]
    top_retrieved_passages: list[str]

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
    facts: list[EntityFact]
    related_entities: list[str]

@dataclass
class ParsedQuestion:
    raw: str
    question_type: str
    entities: list[dict]
    noun_chunks: list[str]
    root_verb: str | None
    sub_questions: list[str]
    keywords: list[str]
```

## Edge Cases

1. **Empty question string** → raises `ValueError` in `WatsonLite.answer()`
2. **No Wikipedia results** → returns `FinalAnswer` with confidence 0.0 and "Could not retrieve relevant passages."
3. **No entities found** → graph enrichment is skipped, confidence relies on extraction + agreement + rank
4. **Extractive model fails on a passage** → that passage is skipped; remaining candidates are used
5. **No candidates after extraction** → `ConfidenceScorer` returns "No answer found" with confidence 0.0
6. **Question with conjunctions** → decomposed into sub-questions; candidates from all sub-questions are merged
7. **Entity name with leading article** → Wikidata lookup strips "the/a/an" before querying
8. **SPARQL/Wikidata API rate limiting** → retries with backoff, falls back to REST API, caches results
9. **HuggingFace model loading failure** → raises `OSError` in `__init__`

## Performance & Constraints

- All inference runs on CPU (no GPU required)
- Pipeline completes in < 30s on cold cache for a typical question
- Wikipedia fetches limited to 5 articles per query
- Cross-encoder reranks top 50 RRF candidates
- Extractive reader processes top 5 ranked passages per sub-question
- Maximum 15 facts per Wikidata entity
- Cache database stored as `watson_lite_cache.sqlite3` in project root
