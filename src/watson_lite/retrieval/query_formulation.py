from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from watson_lite.core.models import ParsedQuestion

logger = logging.getLogger(__name__)

_QUESTION_WORDS = frozenset(
    {"who", "what", "when", "where", "why", "how", "which", "whom", "whose"}
)
_BE_VERBS = frozenset({"is", "are", "was", "were", "be", "been", "am"})
_DO_VERBS = frozenset({"do", "does", "did", "done", "doing"})
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "about", "also", "and", "but", "or", "if", "because", "up", "down",
        "it", "its", "this", "that", "these", "those", "he", "she", "they",
        "we", "you", "me", "him", "her", "them", "my", "your", "his", "their",
        "did", "does", "has", "have", "had", "been", "being",
    }
)


def _content_words(text: str) -> set[str]:
    words = text.lower().split()
    return {
        w
        for w in words
        if w not in _STOPWORDS
        and w not in _QUESTION_WORDS
        and w not in _BE_VERBS
        and w not in _DO_VERBS
        and len(w) > 1
    }


def _entity_to_noun_chunk(
    entity_text: str, noun_chunks: list[str]
) -> str | None:
    entity_lower = entity_text.lower()
    for chunk in noun_chunks:
        if entity_lower in chunk.lower():
            return chunk
    return None


def _augmented_queries(
    parsed: ParsedQuestion,
) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        qs = q.strip()
        if qs and qs not in seen:
            seen.add(qs)
            variants.append(qs)

    _add(parsed.raw)

    entity_texts = [str(e["text"]) for e in parsed.entities]
    # Keyword query: verb + entity names
    if parsed.root_verb and entity_texts:
        _add(f"{parsed.root_verb} {' '.join(entity_texts)}")
    elif parsed.keywords:
        _add(" ".join(parsed.keywords))

    # Entity-only queries — use full noun chunk when possible
    if entity_texts:
        richer_entity_terms: list[str] = []
        for e in entity_texts:
            chunk = _entity_to_noun_chunk(e, parsed.noun_chunks)
            richer_entity_terms.append(chunk if chunk else e)
        _add(" ".join(richer_entity_terms))
    elif parsed.noun_chunks:
        longest = max(parsed.noun_chunks, key=lambda c: len(c))
        _add(longest)

    # Raw + entity as a focused query
    if entity_texts:
        raw_lower = parsed.raw.lower()
        extra = [e for e in entity_texts if e.lower() not in raw_lower]
        if extra:
            _add(f"{parsed.raw} {' '.join(extra)}")

    # LAT + entity query (when LAT is known)
    if parsed.lat and entity_texts:
        _add(f"{parsed.lat} {' '.join(entity_texts)}")

    # Sub-question (if different from original and > 2 words)
    for sq in parsed.sub_questions:
        if sq.lower() != parsed.raw.lower():
            words = sq.split()
            if len(words) > 2:
                _add(sq)

    # Type-specific fallback queries
    if parsed.question_type == "when" and entity_texts:
        _add(f"{' '.join(entity_texts)} date")
        _add(f"{' '.join(entity_texts)} year")

    if parsed.question_type == "where" and entity_texts:
        _add(f"{' '.join(entity_texts)} location")

    if parsed.question_type == "how" and entity_texts:
        _add(f"{' '.join(entity_texts)} how")

    if parsed.question_type == "why" and entity_texts:
        _add(f"{' '.join(entity_texts)} reason")

    return variants[:5]


def _original_queries(parsed: ParsedQuestion) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        qs = q.strip()
        if qs and qs not in seen:
            seen.add(qs)
            variants.append(qs)

    _add(parsed.raw)

    entity_texts = [str(e["text"]) for e in parsed.entities]
    if parsed.root_verb and entity_texts:
        _add(f"{parsed.root_verb} {' '.join(entity_texts)}")
    elif parsed.keywords:
        _add(" ".join(parsed.keywords))

    if entity_texts:
        _add(" ".join(entity_texts))

    if parsed.lat and entity_texts:
        _add(f"{parsed.lat} {' '.join(entity_texts)}")

    for sq in parsed.sub_questions:
        if sq.lower() != parsed.raw.lower():
            words = sq.split()
            if len(words) > 2:
                _add(sq)

    if parsed.question_type == "when" and entity_texts:
        _add(f"{' '.join(entity_texts)} date")
        _add(f"{' '.join(entity_texts)} year")

    if parsed.question_type == "where" and entity_texts:
        _add(f"{' '.join(entity_texts)} location")

    if parsed.question_type == "how" and entity_texts:
        _add(f"{' '.join(entity_texts)} how")

    if parsed.question_type == "why" and entity_texts:
        _add(f"{' '.join(entity_texts)} reason")

    return variants[:5]


def generate_search_queries(
    parsed: ParsedQuestion,
    *,
    augment_context: bool = True,
) -> list[str]:
    """Generate up to 5 search query variants from the parsed question.

    Different phrasings return different Wikipedia articles, increasing
    passage recall.  The first query is always the original question text.

    When *augment_context* is True (default), entity-only queries are enriched
    with their surrounding noun-chunk context and a raw+entity disambiguation
    variant is added.
    """
    if augment_context:
        return _augmented_queries(parsed)
    return _original_queries(parsed)
