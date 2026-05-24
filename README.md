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
git clone https://github.com/daedalus/watson_lite.git
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
