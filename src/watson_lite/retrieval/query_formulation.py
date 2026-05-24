from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from watson_lite.core.models import ParsedQuestion

logger = logging.getLogger(__name__)


def generate_search_queries(parsed: ParsedQuestion) -> list[str]:
    """Generate up to 5 search query variants from the parsed question.

    Different phrasings return different Wikipedia articles, increasing
    passage recall.  The first query is always the original question text.
    """
    variants: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        qs = q.strip()
        if qs and qs not in seen:
            seen.add(qs)
            variants.append(qs)

    _add(parsed.raw)

    # Keyword query: verb + entity names
    entity_texts = [str(e["text"]) for e in parsed.entities]
    if parsed.root_verb and entity_texts:
        _add(f"{parsed.root_verb} {' '.join(entity_texts)}")
    elif parsed.keywords:
        _add(" ".join(parsed.keywords))

    # Entity-only query
    if entity_texts:
        _add(" ".join(entity_texts))

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
