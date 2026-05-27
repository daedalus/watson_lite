from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from watson_lite.core.models import ParsedQuestion

logger = logging.getLogger(__name__)


def _entity_to_noun_chunk(entity_text: str, noun_chunks: list[str]) -> str | None:
    entity_lower = entity_text.lower()
    for chunk in noun_chunks:
        if entity_lower in chunk.lower():
            return chunk
    return None


def _add_variant(q: str, variants: list[str], seen: set[str]) -> None:
    qs = q.strip()
    if qs and qs not in seen:
        seen.add(qs)
        variants.append(qs)


def _add_sub_questions(
    sub_questions: list[str],
    raw: str,
    variants: list[str],
    seen: set[str],
) -> None:
    for sq in sub_questions:
        if sq.lower() != raw.lower():
            words = sq.split()
            if len(words) > 2:
                _add_variant(sq, variants, seen)


_INTERROGATIVE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "when": ("date", "year"),
    "where": ("location",),
    "why": ("reason",),
    "how": ("how",),
}


def _add_type_suffix_queries(
    question_word: str | None,
    entity_texts: list[str],
    variants: list[str],
    seen: set[str],
) -> None:
    entity_str = " ".join(entity_texts)
    for suffix in _INTERROGATIVE_SUFFIXES.get(question_word, ()):
        _add_variant(f"{entity_str} {suffix}", variants, seen)


def _add_entity_enriched_query(
    entity_texts: list[str],
    noun_chunks: list[str],
    variants: list[str],
    seen: set[str],
) -> None:
    richer_entity_terms: list[str] = []
    for e in entity_texts:
        chunk = _entity_to_noun_chunk(e, noun_chunks)
        richer_entity_terms.append(chunk if chunk else e)
    _add_variant(" ".join(richer_entity_terms), variants, seen)


def _add_entity_disambiguation_query(
    raw: str,
    entity_texts: list[str],
    variants: list[str],
    seen: set[str],
) -> None:
    raw_lower = raw.lower()
    extra = [e for e in entity_texts if e.lower() not in raw_lower]
    if extra:
        _add_variant(f"{raw} {' '.join(extra)}", variants, seen)


def _augmented_queries(
    parsed: ParsedQuestion,
) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    _add_variant(parsed.raw, variants, seen)

    entity_texts = [str(e["text"]) for e in parsed.entities]
    if parsed.root_verb and entity_texts:
        _add_variant(f"{parsed.root_verb} {' '.join(entity_texts)}", variants, seen)
    elif parsed.keywords:
        _add_variant(" ".join(parsed.keywords), variants, seen)

    if entity_texts:
        _add_entity_enriched_query(entity_texts, parsed.noun_chunks, variants, seen)
    elif parsed.noun_chunks:
        chunks = list(parsed.noun_chunks)
        if chunks:
            _add_variant(max(chunks, key=len), variants, seen)

    if entity_texts:
        _add_entity_disambiguation_query(parsed.raw, entity_texts, variants, seen)

    if parsed.lat and entity_texts:
        _add_variant(f"{parsed.lat} {' '.join(entity_texts)}", variants, seen)

    _add_sub_questions(parsed.sub_questions, parsed.raw, variants, seen)

    if entity_texts:
        _add_type_suffix_queries(parsed.question_word, entity_texts, variants, seen)

    return variants[:5]


def _original_queries(parsed: ParsedQuestion) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    _add_variant(parsed.raw, variants, seen)

    entity_texts = [str(e["text"]) for e in parsed.entities]
    if parsed.root_verb and entity_texts:
        _add_variant(f"{parsed.root_verb} {' '.join(entity_texts)}", variants, seen)
    elif parsed.keywords:
        _add_variant(" ".join(parsed.keywords), variants, seen)

    if entity_texts:
        _add_variant(" ".join(entity_texts), variants, seen)

    if parsed.lat and entity_texts:
        _add_variant(f"{parsed.lat} {' '.join(entity_texts)}", variants, seen)

    _add_sub_questions(parsed.sub_questions, parsed.raw, variants, seen)

    if entity_texts:
        _add_type_suffix_queries(parsed.question_word, entity_texts, variants, seen)

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
