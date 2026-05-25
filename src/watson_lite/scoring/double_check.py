from __future__ import annotations

import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from watson_lite.retrieval.dataset_query_engine import DatasetQueryEngine

_STOPWORDS = {
    "the",
    "is",
    "are",
    "was",
    "what",
    "who",
    "how",
    "when",
    "where",
    "why",
    "which",
    "did",
    "does",
    "do",
    "a",
    "an",
    "in",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
}


def _question_keywords(question: str) -> list[str]:
    keywords: list[str] = []
    for token in question.split():
        cleaned = token.strip(string.punctuation).lower()
        if len(cleaned) < 3 or cleaned in _STOPWORDS:
            continue
        keywords.append(cleaned)
    return keywords


def bidirectional_score(
    span: str,
    question: str,
    dataset_query_engine: DatasetQueryEngine,
    *,
    top_k: int = 3,
) -> float:
    """Bidirectional answer validation — re-query Wikipedia with the candidate answer."""
    if not span.strip() or top_k <= 0:
        return 0.0

    keywords = _question_keywords(question)
    if not keywords:
        return 0.0

    passages = dataset_query_engine.query(span, top_k=top_k)
    if not passages:
        return 0.0

    matched_passages = 0
    for passage in passages[:top_k]:
        passage_text = passage.text.lower()
        if any(keyword in passage_text for keyword in keywords):
            matched_passages += 1

    return max(0.0, min(matched_passages / top_k, 1.0))
