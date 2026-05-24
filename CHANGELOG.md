# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-24

### Added
- Initial release — extractive QA pipeline with no LLM and no training
- NLP preprocessing via spaCy (NER, question classification, decomposition)
- Parallel retrieval: BM25 + dense vector (sentence-transformers + FAISS)
- Graph enrichment: Wikidata entity lookup + fact retrieval
- RRF fusion + cross-encoder re-ranking
- Extractive QA with deepset/roberta-base-squad2
- Multi-signal confidence scoring
- SQLite3 cache for Wikipedia and Wikidata responses
- Interactive CLI and single-shot CLI modes

[0.1.0]: https://github.com/daedalus/watson_lite/releases/tag/v0.1.0
