from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from watson_lite.core.models import Passage, RankedPassage


_WORD_RE = re.compile(r"[a-z]+")

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "it",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "any",
        "because",
        "before",
        "between",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "into",
        "over",
        "under",
        "up",
        "out",
        "off",
        "down",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
    }
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOP_WORDS]


def _compute_idf(passages: list[Passage]) -> dict[str, float]:
    n = len(passages)
    if n == 0:
        return {}
    df: dict[str, int] = {}
    for p in passages:
        for tok in set(_tokenize(p.text)):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log(1 + n / freq) for tok, freq in df.items()}


def score_term_match(
    question: str,
    ranked_passages: list[RankedPassage],
) -> float:
    if not ranked_passages:
        return 0.0
    passages = [rp.passage for rp in ranked_passages]
    idf = _compute_idf(passages)
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0

    best_score = 0.0
    for rp in ranked_passages:
        p_tokens = set(_tokenize(rp.passage.text))
        num = 0.0
        denom = 0.0
        for tok in q_tokens:
            w = idf.get(tok, 1.0)
            denom += w
            if tok in p_tokens:
                num += w
        score = num / denom if denom > 0 else 0.0
        best_score = max(best_score, score)

    return best_score
