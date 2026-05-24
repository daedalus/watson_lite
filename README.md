# watson-lite

A Watson-inspired extractive QA system that runs on a laptop.  
**No LLM. No trained weights of your own. No paid APIs.**

[![Python](https://img.shields.io/pypi/pyversions/watson-lite.svg)](https://pypi.org/project/watson-lite/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/master/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Install

```bash
pip install watson-lite
python -m spacy download en_core_web_sm
```

## Usage

### CLI

```bash
# Single question
watson-lite "Who designed the Eiffel Tower?"
watson-lite "Who was the 44th president of the United States?"

# Interactive mode
watson-lite
```

### Python

```python
from watson_lite import WatsonLite

watson = WatsonLite()
answer = watson.answer("Who designed the Eiffel Tower?")

print(answer.answer)        # "Gustave Eiffel"
print(answer.confidence)    # 0.847
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
- **`BM25Retriever`** — BM25 retrieval over Wikipedia REST API.
- **`VectorRetriever`** — Dense vector retrieval (sentence-transformers + FAISS).
- **`WikidataGraph`** — Structured fact enrichment from Wikidata.
- **`Ranker`** — RRF fusion + cross-encoder re-ranking.
- **`ExtractiveReader`** — Span extraction via roberta-base-squad2.
- **`ConfidenceScorer`** — Multi-signal confidence scoring.
- **`Cache`** — SQLite3 cache for Wikipedia and Wikidata responses.

## Development

```bash
git clone https://github.com/daedalus/watson-lite.git
cd watson_lite
pip install -e ".[test]"

# run tests
pytest

# format
ruff format src/ tests/

# lint + type check
prospector --with-tool ruff --with-tool mypy src/

# find unused code
vulture --min-confidence 90 src/
```

## Architecture

```
User Question → NLP (spaCy) → Decomposition → Entity Extraction
  → Parallel Retrieval (BM25 + FAISS) → Graph (Wikidata)
  → RRF Fusion → Cross-Encoder Rerank → Span Extraction → Confidence Score
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
- **Wikidata REST API** — Structured entity facts (no SPARQL)

## Extending

- **Add a domain corpus**: Replace `fetch_wikipedia_passages()` with your own document loader.
- **Add more graph sources**: Wikidata REST API pattern is reusable.
- **Offline mode**: Download Wikipedia dumps and index locally with BM25 + FAISS.
