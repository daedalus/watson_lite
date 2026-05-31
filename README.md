# watson-lite

A Watson-inspired extractive QA system that runs on a laptop.  
**No LLM. No trained weights of your own. No paid APIs.**

[![Python](https://img.shields.io/pypi/pyversions/watson-lite.svg)](https://pypi.org/project/watson-lite/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/master/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Why?

LLMs have largely displaced classical QA systems, but that displacement came
with trade-offs that matter in practice. watson-lite optimises for the
constraints LLMs handle poorly.

**Transparency and auditability.** Every step in the pipeline is a named,
inspectable object: BM25 scores, RRF fusion weights, span extraction logits,
graph corroboration flags, calibrated confidence breakdown. You can trace
exactly *why* the answer is what it is. For regulated industries, research
workflows, or anywhere an answer must be explained and verified, that audit
trail is essential.

**No hallucination by construction.** Extractive QA cannot invent a fact that
is not in a retrieved passage. Every answer is a verbatim span from a public
source. The confidence score is a function of extraction score, span agreement
across independent retrievals, graph corroboration, and rank signal — all
measurable and falsifiable. A language model can be confidently wrong;
watson-lite either finds a span or returns confidence 0.

**Cost and data sovereignty.** Zero token cost, zero API keys, no data sent to
a third party. The system runs on a laptop CPU with roughly 670 MB of
pretrained models and a live REST connection to Wikipedia — public
infrastructure. Anyone who cannot or will not pay per-query or expose queries
to a vendor can run this as-is.

**A teaching object for classical NLP architecture.** The
[GAP analysis](GAP.md) traces every design decision back to the Ferrucci et
al. DeepQA papers. Reading the codebase alongside those papers is a
curriculum: question analysis → retrieval → hypothesis generation → scoring →
confidence calibration. That pipeline is still the skeleton inside every
modern RAG system; watson-lite makes the skeleton visible and runnable.

**A composable RAG building block.** Once a persistent BM25/FAISS index is
wired to a domain corpus (see GAP-01 in [GAP.md](GAP.md)), watson-lite becomes
a deterministic retriever and reader that can feed a generation layer or stand
alone. The design question is not *"does this beat GPT-4?"* but *"what do you
get when you optimise for explainability, zero hallucination, and zero cost
instead of raw benchmark score?"*


**Caveat.** Even though this system gives factual correct answers it might give not the expected answer due to question interpretability and or lack of context.


## Install

```bash
pip install "watson-lite[full]"
python -m spacy download en_core_web_sm
```

Optional extras:

- `watson-lite[nlp]` — spaCy question processing
- `watson-lite[vector]` — dense retrieval dependencies
- `watson-lite[rerank]` — cross-encoder reranking
- `watson-lite[reader]` — extractive QA reader
- `watson-lite[graph]` — SPARQL fallback support
- `watson-lite[full]` — all runtime features

## Usage

### CLI

```bash
# Single question
watson-lite "Who designed the Eiffel Tower?"
watson-lite "Who was the 44th president of the United States?"

# Interactive mode
watson-lite

# Minimal profile + JSON output
watson-lite --profile minimal --output json "Who designed the Eiffel Tower?"

# Clear cache before running
watson-lite --clear-cache "Who designed the Eiffel Tower?"

# Toggle optional features (ablation-style)
watson-lite --no-vector-retrieval --no-graph-enrichment "Who designed the Eiffel Tower?"

# Query across multiple online datasets
watson-lite --datasets wikipedia,wikibooks "What is Python?"

# Query additional public sources
watson-lite --datasets wikiquote,wikisource,wikinews,pubmed,arxiv "What is Python?"

# Query an offline corpus plugin counterpart
watson-lite \
  --datasets wikipedia_offline \
  --offline-dataset-dir /path/to/offline-corpora \
  "What is Python?"

# Query Elasticsearch
watson-lite \
  --datasets elasticsearch \
  --elasticsearch-url http://localhost:9200 \
  --elasticsearch-index wiki_passages \
  "What is Python?"

# Query Hugging Face datasets-server
watson-lite \
  --datasets huggingface \
  --huggingface-dataset ag_news \
  --huggingface-config default \
  --huggingface-split train \
  "What is Python?"

# Benchmark/eval run from dataset
watson-lite \
  --benchmark-dataset /path/to/benchmark.json \
  --benchmark-output-json /tmp/watson_benchmark.json \
  --benchmark-output-csv /tmp/watson_benchmark.csv

# Full ablation sweep + regression gate against baseline
watson-lite \
  --benchmark-dataset /path/to/benchmark.json \
  --ablation-sweep \
  --regression-check \
  --max-accuracy-drop 0.02 \
  --max-f1-drop 0.02

# Plugin management commands
watson-lite plugins list
watson-lite plugins list --mode offline
watson-lite plugins describe wikipedia
watson-lite plugins validate --datasets wikipedia,wikipedia_offline
```

Benchmark dataset format (`.json` or `.jsonl`):

```json
[
  {
    "question": "Who designed the Eiffel Tower?",
    "answers": ["Gustave Eiffel"],
    "evidence_passages": ["designed by Gustave Eiffel"]
  }
]
```

### Python

```python
from watson_lite import WatsonLite

watson = WatsonLite()
answer = watson.answer("Who designed the Eiffel Tower?")

print(answer.answer)        # "Gustave Eiffel"
print(answer.confidence)    # 0.752
print(answer.source)        # "Eiffel Tower"
```

### KPI evaluation

```python
from watson_lite import WatsonLite
from watson_lite.evaluation import BenchmarkLabel, evaluate_kpis

watson = WatsonLite()
answers = [
    watson.answer("Who designed the Eiffel Tower?", verbose=False),
    watson.answer("What is the capital of France?", verbose=False),
]

labels = [
    BenchmarkLabel(
        answers=["Gustave Eiffel"],
        evidence_passages=["designed by Gustave Eiffel"],
    ),
    BenchmarkLabel(
        answers=["Paris"],
        evidence_passages=["capital of France"],
    ),
]

report = evaluate_kpis(answers, labels, recall_k=10, calibration_bins=10)
print(report.answer_success_rate)
print(report.latency_p95_s)
print(report.confidence_calibration_ece)
print(report.confidence_calibration_kl_divergence)
print(report.confidence_calibration_js_divergence)
```

Each `FinalAnswer` now includes `diagnostics` with stage latencies, cache hit/miss
counters, retrieval/extraction counts, and top retrieved passages for KPI rollups.

### Example output

```
$ watson-lite "Who was the 44th president of the United States?"

  ANSWER:     Barack Hussein Obama
  CONFIDENCE: 43.6%
  SOURCE:     Barack Obama
  URL:        https://en.wikipedia.org/wiki/Barack Obama

  Confidence breakdown:
    extraction_model: 0.592
    span_agreement: 0.2
    graph_corroboration: 0.0
    passage_rank_signal: 1.0

  Time: 44.60s
```

## API

- **`WatsonLite`** — Main orchestrator. `answer(question)` runs the full 6-stage pipeline.
- **`NLPProcessor`** — spaCy-based question classification, NER, decomposition.
- **`DatasetQueryEngine`** — Modular dataset querying and aggregation across pluggable providers.
- **`BM25Retriever`** — BM25 retrieval over aggregated online passages.
- **`VectorRetriever`** — Dense vector retrieval (sentence-transformers + FAISS).
- **`WikidataGraph`** — Structured fact enrichment from Wikidata.
- **`Ranker`** — RRF fusion + cross-encoder re-ranking.
- **`ExtractiveReader`** — Span extraction via roberta-base-squad2.
- **`ConfidenceScorer`** — Multi-signal confidence scoring.
- **`Cache`** — SQLite3 cache for Wikipedia/Wikidata/type-coercion responses with TTL expiry, namespace metrics, and bounded-size pruning.

## Feature inventory

Core (always on):
- NLP parse
- Dataset query engine fetch
- BM25 retrieve
- Span extraction
- Final scoring shell

Optional toggles (default enabled):
- Vector retrieval (`--no-vector-retrieval`)
- Query expansion variants (`--no-query-expansion`)
- Wikidata graph enrichment (`--no-graph-enrichment`)
- Cross-encoder reranking (`--no-cross-encoder-reranking`)
- Question-type bonus (`--no-question-type-bonus`)
- Type-coercion signal (`--no-type-coercion`)

Dataset providers:
- `wikipedia`
- `wikibooks`
- `wikiquote`
- `wikisource`
- `wikinews`
- `pubmed`
- `arxiv`
- `openlibrary`
- `stackexchange`
- `dbpedia`
- `oeis`
- `elasticsearch` (configure with `--elasticsearch-url` and `--elasticsearch-index`, or `WATSON_LITE_ELASTICSEARCH_URL` and `WATSON_LITE_ELASTICSEARCH_INDEX`)
- `huggingface`
  - required: `--huggingface-dataset`, `--huggingface-split`
  - optional: `--huggingface-config`, `--huggingface-token`
  - env vars: `WATSON_LITE_HUGGINGFACE_DATASET`, `WATSON_LITE_HUGGINGFACE_SPLIT`, `WATSON_LITE_HUGGINGFACE_CONFIG`, `WATSON_LITE_HUGGINGFACE_TOKEN`

Offline counterpart plugins:
- every built-in online dataset plugin has a matching `*_offline` plugin
  (`wikipedia_offline`, `pubmed_offline`, `huggingface_offline`, etc.)
- each offline plugin reads local JSON/JSONL from:
  - `--offline-dataset-dir /path/to/corpora` + `<dataset>.jsonl`
  - or env var `WATSON_LITE_OFFLINE_<DATASET>_PATH`
  - or env var `WATSON_LITE_OFFLINE_DATASET_DIR`

## Development

```bash
git clone https://github.com/daedalus/watson-lite.git
cd watson_lite
pip install -e ".[test,lint,full]"

# run tests
pytest

# format
ruff format src/ tests/

# lint + type check
prospector --with-tool ruff --with-tool mypy src/

# find unused code
vulture --min-confidence 90 src/
```

Checked-in benchmark smoke dataset: `benchmarks/smoke.json`

## Architecture

```text
                          +------------------+
                          |  User question   |
                          +------------------+
                                    |
                                    v
                    +-----------------------------+        +-----------------------------+
                    | NLPProcessor                |------->| WikidataGraph               |
                    | - classify question         | entity | - enrich extracted entities |
                    | - extract entities/keywords | names  +-----------------------------+
                    +-----------------------------+                    | graph_results
                          |          |                                 |
                    queries|  sub-    |                                 |
                           | questions|                                 |
                           v          v                                 |
                    +--------------------------+                        |
                    | Query expansion +        |                        |
                    | sub-questions            |                        |
                    +--------------------------+                        |
                                    |                                   |
                                    v                                   |
                    +-----------------------------+                     |
                    | DatasetQueryEngine          |                     |
                    | - Wikipedia REST API        |                     |
                    | - Wikibooks REST API        |                     |
                    +-----------------------------+                     |
                                    |                                   |
                                    v                                   |
                    +-----------------------------+                     |
                    | Parallel retrieval          |                     |
                    | - BM25Retriever             |                     |
                    | - VectorRetriever (FAISS)   |                     |
                    +-----------------------------+                     |
                                    |                                   |
                                    v                                   |
                    +-----------------------------+                     |
                    | Ranker                      |                     |
                    | - RRF fusion                |                     |
                    | - cross-encoder rerank      |                     |
                    +-----------------------------+                     |
                                    |                                   |
                                    v                                   |
                    +-----------------------------+                     |
                    | ExtractiveReader            |                     |
                    | - answer span extraction    |                     |
                    +-----------------------------+                     |
                                    |                                   |
                                    +-------------------+---------------+
                                                        |
                                                        v
                                        +-----------------------------+
                                        | ConfidenceScorer            |
                                        | - extraction score          |
                                        | - span agreement            |
                                        | - graph corroboration       |
                                        | - rank / type-coercion      |
                                        +-----------------------------+
                                                        |
                                                        v
                                        +-----------------------------+
                                        | FinalAnswer + diagnostics   |
                                        +-----------------------------+

SQLite cache (cross-cutting): backs DatasetQueryEngine fetches,
Wikidata lookups, and type-coercion calls with TTL-expiry entries.
```

## Models Used (all pretrained, inference only)

| Model | Purpose | Size |
|---|---|---|
| `en_core_web_sm` | spaCy NLP | ~12MB |
| `all-MiniLM-L6-v2` | Passage embeddings | ~90MB |
| `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking | ~90MB |
| `deepset/roberta-base-squad2` | Extractive span QA | ~480MB |

Total: ~670MB — runs CPU-only.

## Data Sources

- **Wikipedia REST API** — Live article retrieval
- **Wikibooks REST API** — Live educational content retrieval
- **Wikidata REST API** — Structured entity facts (no SPARQL)

## Extending

### Dataset retriever plugins

watson-lite loads dataset retrievers from built-ins and Python entry points in
the `watson_lite.dataset_retrievers` group.

Use the CLI to inspect what is currently available:

```bash
watson-lite plugins list
watson-lite plugins describe wikipedia
```

Plugin contract:

- export a `DatasetRetrieverPlugin` instance, or a callable returning one (or a tuple/list of them)
- implement `fetcher(query: str, *, top_k: int) -> list[Passage]`
- set a stable plugin `name`, `mode` (`online` or `offline`), and `description`

Minimal package example (`pyproject.toml`):

```toml
[project.entry-points."watson_lite.dataset_retrievers"]
my_domain = "my_package.my_plugins:build_plugins"
```

`my_package/my_plugins.py` should return plugin objects using
`watson_lite.retrieval.dataset_plugins.DatasetRetrieverPlugin`.

### Offline plugin datasets

Built-in `*_offline` plugins read local JSON/JSONL files. Recommended JSONL row
fields:

- `text` (required)
- `source` (optional)
- `url` (optional)

### Other extensions

- Add more graph sources: Wikidata REST API pattern is reusable.

## Citation

@misc{watson_lite,
  author = {Darío Clavijo},
  title = {watson_lite},
  year = {2026},
  url = {https://github.com/daedalus/watson_lite/releases/tag/v0.1.3},
  note = {Version 0.1.3}
}
