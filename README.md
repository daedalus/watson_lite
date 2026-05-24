# WatsonLite

A Watson-inspired extractive QA system that runs on a laptop.  
**No LLM. No trained weights of your own. No paid APIs.**

---

## Architecture

```
User Question
      ↓
NLP Preprocessing         (spaCy — NER, POS, dependency parse, question type)
      ↓
Rule-Based Decomposition  (conjunctions → sub-questions)
      ↓
Entity Extraction         (spaCy NER → Wikidata entity linking)
      ↓
┌─────────────────────────────────┐
│  PARALLEL RETRIEVAL             │
│  BM25 (Wikipedia REST API)      │
│  Vector (FAISS + MiniLM embeds) │
└────────────────┬────────────────┘
                 ↓
Candidate Extraction      (merge + dedup)
                 ↓
Graph Enrichment          (Wikidata SPARQL — free, no key)
                 ↓
RRF Fusion                (Reciprocal Rank Fusion — no training)
                 ↓
Cross-Encoder Rerank      (ms-marco-MiniLM — pretrained, inference only)
                 ↓
Multi-Hypothesis Extraction (roberta-base-squad2 — span extraction, no generation)
                 ↓
Confidence Scoring        (extraction score + agreement + graph corroboration)
                 ↓
Final Answer + Confidence Score
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Usage

### Interactive CLI
```bash
python cli.py
```

### Programmatic
```python
from pipeline import WatsonLite

watson = WatsonLite()
answer = watson.answer("Who designed the Eiffel Tower?")

print(answer.answer)        # "Gustave Eiffel"
print(answer.confidence)    # 0.847
print(answer.source)        # "Eiffel Tower"
print(answer.graph_facts)   # ["architect: Gustave Eiffel", ...]
```

### Run example questions
```bash
python pipeline.py
```

---

## Project Structure

```
watson_lite/
├── requirements.txt
├── pipeline.py          ← main orchestrator
├── cli.py               ← interactive CLI
├── core/
│   ├── nlp.py           ← spaCy preprocessing + question classification
│   └── extractor.py     ← extractive QA + confidence scoring
├── retrieval/
│   ├── bm25_retriever.py   ← BM25 over Wikipedia REST API
│   └── vector_retriever.py ← FAISS + sentence-transformers
├── graph/
│   └── wikidata.py      ← Wikidata SPARQL enrichment
└── ranking/
    └── ranker.py        ← RRF fusion + cross-encoder reranking
```

---

## Models Used (all pretrained, inference only)

| Model | Purpose | Size |
|---|---|---|
| `en_core_web_sm` | spaCy NLP | ~12MB |
| `all-MiniLM-L6-v2` | Passage embeddings | ~90MB |
| `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking | ~90MB |
| `deepset/roberta-base-squad2` | Extractive span QA | ~480MB |

Total: ~670MB downloaded once, runs CPU-only.

---

## Data Sources (all free, no API key)

| Source | Use |
|---|---|
| Wikipedia REST API | Live article retrieval |
| Wikidata SPARQL | Structured entity facts |

---

## Extending

- **Add a domain corpus**: Replace `fetch_wikipedia_passages()` with your own document loader and re-index with FAISS.
- **Add more graph sources**: DBpedia SPARQL endpoint uses the same `SPARQLWrapper` pattern.
- **Offline mode**: Download Wikipedia dumps and index locally with BM25s + FAISS for fully offline operation.
