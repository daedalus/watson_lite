from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from watson_lite.core.models import AnswerCandidate, GraphResult


_YEAR = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")

_TEMPORAL_LABELS: frozenset[str] = frozenset(
    {
        "date of birth",
        "date of death",
        "inception",
        "publication date",
        "dissolved",
        "point in time",
        "date of publication",
    }
)

_GEOSPATIAL_LABELS: frozenset[str] = frozenset(
    {
        "country",
        "located in",
        "location",
        "continent",
        "place of birth",
        "place of death",
        "capital",
        "country of origin",
        "country of citizenship",
        "headquarters",
    }
)


def _extract_years(text: str) -> set[str]:
    return set(_YEAR.findall(text))


def score_temporal_consistency(
    candidates: list[AnswerCandidate],
    graph_results: list[GraphResult],
) -> float:
    if not candidates or not graph_results:
        return 0.0

    best = candidates[0]
    span_years = _extract_years(best.span)
    if not span_years:
        return 0.0

    temporal_facts: list[str] = []
    for gr in graph_results:
        for fact in gr.facts:
            if fact.property_label in _TEMPORAL_LABELS:
                temporal_facts.append(fact.value)

    if not temporal_facts:
        return 0.0

    for fact_val in temporal_facts:
        fact_years = _extract_years(fact_val)
        if span_years & fact_years:
            return 1.0

    return 0.0


def score_geospatial_consistency(
    candidates: list[AnswerCandidate],
    graph_results: list[GraphResult],
) -> float:
    if not candidates or not graph_results:
        return 0.0

    best = candidates[0].span.lower().strip()
    if not best:
        return 0.0

    geo_facts: list[str] = []
    for gr in graph_results:
        for fact in gr.facts:
            if fact.property_label in _GEOSPATIAL_LABELS:
                geo_facts.append(fact.value)

    if not geo_facts:
        return 0.0

    matches = 0
    for fact_val in geo_facts:
        fv = fact_val.lower().strip()
        if best == fv or best in fv or fv in best:
            matches += 1

    if matches > 0:
        return min(1.0, matches / max(len(geo_facts), 1) * 2)

    return 0.0
