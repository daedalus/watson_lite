# Gap Analysis: watson-lite vs. IBM DeepQA Papers

This document tracks gaps between the current watson-lite implementation and the
three foundational IBM DeepQA/Watson research papers listed below.  Each gap
entry notes the source paper, its severity, and the relevant part of the
codebase.

**Reference papers**

1. Ferrucci et al. (2010) — *Building Watson: An Overview of the DeepQA Project*
2. Epstein et al. — *Making Watson Fast*
3. Ferrucci et al. (2013) — *Watson: Beyond Jeopardy!*

---

## What watson-lite already covers

| DeepQA capability | watson-lite implementation |
|---|---|
| Question analysis (type, NER, LAT, sub-questions) | `NLPProcessor` in `core/nlp.py` |
| Query expansion variants | `retrieval/query_formulation.py` |
| Hybrid retrieval over unstructured text | `BM25Retriever` + `VectorRetriever`, parallel via `ThreadPoolExecutor` |
| Structured knowledge enrichment | `WikidataGraph` (Wikidata REST API) |
| Candidate scoring — extraction, span agreement, graph, rank, type coercion | `ConfidenceScorer` in `core/extractor.py` |
| Lexical Answer Type + Wikidata type-hierarchy coercion | `scoring/type_coercion.py`, `LAT_QID_MAP` in `core/nlp.py` |
| Answer normalization / merging | `scoring/answer_merging.py` |
| Temporal + geospatial consistency signals | `scoring/consistency.py` |
| Per-stage latency tracking | `stage_latencies` dict in `pipeline.py` |
| Lazy model loading | `_get_nlp`, `_get_vector`, `_get_reader`, etc. in `pipeline.py` |
| SQLite cache with TTL | `core/cache.py` |
| Ablation via feature toggles | `FeatureConfig` in `core/config.py` |
| Evaluation / benchmark framework | `evaluation/kpis.py`, `evaluation/benchmark_runner.py` |

---

## Gaps

### High severity

#### GAP-01 — No offline indexed corpus (Paper 2)

**What DeepQA does:** Watson queried a locally indexed 4 TB Lucene/Indri corpus
in milliseconds.

**What watson-lite does:** Every call hits the live Wikipedia REST API and
re-indexes BM25 in memory (short-circuited only when passages are identical to
the previous call via `_last_passage_hash`).  No persistent BM25 or FAISS index
exists on disk.

**Effect:** Dominant latency source — typical wall-clock time is 40+ seconds,
vs. < 3 s for DeepQA.

**Where to fix:** `retrieval/bm25_retriever.py`, `retrieval/vector_retriever.py`,
add an `offline_index.py` module and a CLI sub-command to build the index from a
Wikipedia dump.

---

#### GAP-02 — Single hypothesis generator (Paper 1)

**What DeepQA does:** Ran dozens of independent candidate generators in
parallel — Wikipedia title matching, infobox field extraction, alias/redirect
expansion, pattern matching over named entities, structured DB lookups, synonym
expansion.  Each generator produced its own candidates, which were then scored
and merged.

**What watson-lite does:** A single generator: the extractive reader applied to
retrieved passages.  There is no title-match generator (candidate = Wikipedia
article title), no infobox extractor, no alias/redirect expansion.

**Where to fix:** `pipeline.py` — introduce a `CandidateGenerator` abstraction
and register multiple implementations.

---

#### GAP-03 — ML-trained final scorer (Paper 1)

**What DeepQA does:** Trained a logistic regression / SVM over 50+ engineered
features on labeled QA pairs, producing per-feature weights learned from data.

**What watson-lite does:** Hand-tuned fixed linear weights in `ConfidenceScorer`
(`0.35 * extraction_conf + 0.10 * agreement + …`).  No training loop, no feature
weight learning, no calibration from data.

**Where to fix:** `core/extractor.py` / new `scoring/train_scorer.py` — add a
`fit(samples)` method to `ConfidenceScorer` that learns weights via scikit-learn
logistic regression.

---

### Medium severity

#### GAP-04 — No per-candidate evidence re-retrieval (Paper 1)

**What DeepQA does:** For each candidate answer, performs a second retrieval pass
— querying the corpus for passages that contain both question keywords *and* the
candidate — to gather supporting and contradicting evidence independently.

**What watson-lite does:** Single retrieval pass for the question only.  Span
agreement across passages is a proxy but does not accumulate cross-document
occurrence counts per candidate.

**Where to fix:** `pipeline.py` `answer()` — after extraction, for each unique
candidate span issue a follow-up `DatasetQueryEngine.query(span + question_keywords)`.

---

#### GAP-05 — No answer re-querying / bidirectional validation (Paper 1)

**What DeepQA does:** Uses the candidate answer as a query and checks whether the
retrieved results mention the original question context ("double-check").

**What watson-lite does:** Not implemented.

**Where to fix:** New `scoring/double_check.py` — take top candidate spans, query
Wikipedia for each, check for question-term overlap in returned passages, return a
signal in [0, 1].

---

#### GAP-06 — Parallel hypothesis scoring (Paper 2)

**What DeepQA does:** Every scoring component ran simultaneously across all
candidates on thousands of cores.

**What watson-lite does:** Retrieval is parallelized (2 threads).  Extraction over
sub-questions, graph enrichment, ranking, and scoring are all sequential.

**Where to fix:** `pipeline.py` `_retrieve_parallel` — extend to also dispatch
graph enrichment and (optionally) per-passage extraction in parallel.

---

#### GAP-07 — Shallow NLP: no SRL or coreference (Paper 1)

**What DeepQA does:** Used semantic role labeling (SRL), full dependency parsing,
and coreference resolution to understand question structure and passage arguments.

**What watson-lite does:** `en_core_web_sm` for shallow NER and noun chunks, plus
root-verb extraction.  No SRL, no coreference.

**Where to fix:** `core/nlp.py` — optionally load spaCy's `en_core_web_trf` or
add an `srl` extra using `allennlp-models` for SRL.

---

#### GAP-08 — No textual entailment component (Paper 1)

**What DeepQA does:** A textual entailment scorer checked whether a passage
logically implies that the candidate is the correct answer to the question.

**What watson-lite does:** The cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
measures passage–question relevance, which is the closest proxy, but it does not
check passage → (candidate is the answer) entailment.

**Where to fix:** New `scoring/entailment.py` — use a pretrained NLI model (e.g.,
`cross-encoder/nli-deberta-v3-small`) to score `passage ⊨ "The answer to '<q>' is '<span>'"`.

---

#### GAP-09 — Limited source diversity (Papers 1, 3)

**What DeepQA does:** 20+ corpora including encyclopedia, newswire, Gutenberg
books, DBpedia, WordNet, and domain-specific databases.

**What watson-lite does:** Wikipedia + Wikibooks REST APIs are wired in by
default; Wikidata for structured facts.  `DatasetQueryEngine` is pluggable, but
no additional providers are bundled.

**Where to fix:** `retrieval/dataset_query_engine.py` — add provider
implementations for Wikisource, OpenLibrary, and simple DBpedia SPARQL queries.

---

#### GAP-10 — No learned confidence threshold / abstention (Paper 3)

**What DeepQA does:** Learned a calibrated confidence threshold from training
data; the system abstained when its top score was below the threshold (the
Jeopardy! "buzzer" decision).

**What watson-lite does:** Always returns an answer with a raw confidence score.
ECE calibration is measured in `evaluation/kpis.py` but is not used to gate
output.

**Where to fix:** `core/extractor.py` `ConfidenceScorer` — add an optional
`threshold` field; return `"I don't know"` (or raise a distinct exception) when
confidence is below it.  Expose `--confidence-threshold` in the CLI.

---

### Low severity

#### GAP-11 — Domain-specific ontologies (Paper 3)

**What DeepQA does:** Plugged in UMLS, SNOMED, ICD-10, FinancialOntology, etc.,
enabling type coercion and entity disambiguation in specialized domains.

**What watson-lite does:** General-purpose `LAT_QID_MAP` with ~30 Wikidata QID
mappings; no domain ontology layer.

**Where to fix:** `core/nlp.py` `LAT_QID_MAP` — make it injectable (e.g., via
`FeatureConfig`) so callers can supply custom LAT → QID mappings.

---

#### GAP-12 — No structured explanation / evidence chain (Paper 3)

**What DeepQA does:** Returned a ranked evidence chain with passage citations,
sentence-level grounding, and property links between the candidate and the answer
type.

**What watson-lite does:** `FinalAnswer.supporting_passages` (raw text slices) and
`graph_facts` (property: value strings) — no structured span→passage→property
chain.

**Where to fix:** `core/models.py` — add an `EvidenceItem` dataclass (passage,
sentence, span offset, graph property); populate it in `ConfidenceScorer.score`.

---

#### GAP-13 — Single-pass retrieval; no iterative re-query (Paper 3)

**What DeepQA does:** Multi-pass loops — retrieve, score, identify low-confidence
areas, re-query with refined terms.

**What watson-lite does:** Strictly single-pass: one retrieval → one ranking →
one extraction, no feedback loop.

**Where to fix:** `pipeline.py` `answer()` — after extraction, if `best.extraction_score < threshold`, generate a follow-up query from the partial answer and re-retrieve.

---

#### GAP-14 — No UIMA-style dataflow scheduling (Paper 2)

**What DeepQA does:** Used Apache UIMA so pipeline components declared their
input/output types; the scheduler dispatched them automatically as dependencies
were satisfied, enabling fine-grained parallelism.

**What watson-lite does:** Imperative sequential Python with manual
`ThreadPoolExecutor` for two retrieval threads.

**Where to fix:** Low priority — the current sequential layout is clear and
sufficient for CPU-only inference.  A dataflow scheduler would only be justified
if many more parallel components were added.

---

#### GAP-15 — No learning from feedback (Paper 3)

**What DeepQA does:** Could be retrained on domain-specific labeled data and user
corrections to improve feature weights.

**What watson-lite does:** No fine-tuning, no feedback logging, no mechanism to
improve from past answers.

**Where to fix:** `evaluation/benchmark_runner.py` — log (question, answer,
correct) triples; add a `train_from_log` utility that feeds them into the
`ConfidenceScorer` training path from GAP-03.

---

## Priority order for implementation

| Priority | Gap | Effort |
|---|---|---|
| 1 | GAP-01 offline index | High |
| 2 | GAP-03 learned scorer | Medium |
| 3 | GAP-02 multiple hypothesis generators | High |
| 4 | GAP-04 per-candidate re-retrieval | Medium |
| 5 | GAP-10 confidence threshold / abstention | Low |
| 6 | GAP-05 bidirectional validation | Medium |
| 7 | GAP-06 parallel hypothesis scoring | Low |
| 8 | GAP-08 textual entailment | Medium |
| 9 | GAP-07 richer NLP (SRL, coref) | High |
| 10 | GAP-09 source diversity | Low |
| 11 | GAP-11 domain ontologies | Low |
| 12 | GAP-12 structured explanation | Low |
| 13 | GAP-13 iterative re-query | Low |
| 14 | GAP-14 UIMA dataflow | Low |
| 15 | GAP-15 learning from feedback | Low |
