"""Wikidata SPARQL dataset retriever — fetches structured facts as passages."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from watson_lite.core.cache import get_cache, is_cache_miss
from watson_lite.core.models import Passage
from watson_lite.core.network import request_json

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_SPARQL_ENDPOINT_ENV = "WATSON_LITE_WIKIDATA_SPARQL_ENDPOINT"
_SPARQL_TIMEOUT_SECONDS = 30
_NEGATIVE_CACHE_TTL_SECONDS = 300


def _format_wikidata_sparql_passage(
    binding: dict[str, Any],
    passages: list[Passage],
    seen_chunks: set[str],
) -> None:
    label_obj = binding.get("itemLabel", {})
    desc_obj = binding.get("description", {})
    entity_obj = binding.get("item", {})

    label = label_obj.get("value", "")
    description = desc_obj.get("value", "")
    entity_uri = entity_obj.get("value", "")

    if not label:
        return

    text = f"{label}: {description}" if description else label
    dedup_key = text.lower().strip()
    if dedup_key in seen_chunks:
        return
    seen_chunks.add(dedup_key)

    url = entity_uri if entity_uri else f"https://www.wikidata.org/wiki/{label}"
    passages.append(Passage(text=text, source="Wikidata", url=url))


def fetch_wikidata_sparql_passages(
    query: str,
    *,
    top_k: int = 5,
    endpoint: str | None = None,
) -> list[Passage]:
    """Fetch passages from Wikidata via SPARQL entity+description queries.

    Returns matching entities with their English labels and descriptions
    as Passage objects.
    """
    cache = get_cache()
    normalized_query = query.lower().strip()
    cache_key = f"wikidata_sparql:passages:{normalized_query}:top_k={top_k}"
    cached = cache.get_or_sentinel(cache_key)
    if not is_cache_miss(cached):
        logger.debug("Cache hit: %s", cache_key)
        return [Passage(**p) for p in cached]

    resolved_endpoint = (
        endpoint
        or os.getenv(WIKIDATA_SPARQL_ENDPOINT_ENV)
        or WIKIDATA_SPARQL_ENDPOINT
    )
    capped_top_k = min(top_k, 50)

    safe_query = re.sub(r'["\'\\]', " ", normalized_query).strip()
    sparql = (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        "PREFIX schema: <http://schema.org/> "
        "SELECT ?item ?itemLabel ?description WHERE { "
        "  ?item rdfs:label ?itemLabel . "
        "  ?item schema:description ?description . "
        "  FILTER(lang(?itemLabel) = 'en') "
        "  FILTER(lang(?description) = 'en') "
        f'  FILTER(CONTAINS(LCASE(?itemLabel), "{safe_query}")) '
        f"}} LIMIT {capped_top_k}"
    )

    payload = request_json(
        resolved_endpoint,
        params={"query": sparql, "format": "application/sparql-results+json"},
        timeout=_SPARQL_TIMEOUT_SECONDS,
        context="Wikidata SPARQL",
    )
    if payload is None:
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    bindings = (
        payload.get("results", {}).get("bindings", [])
        if isinstance(payload, dict)
        else []
    )
    if not isinstance(bindings, list):
        cache.set(cache_key, [], ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
        return []

    passages: list[Passage] = []
    seen_chunks: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        _format_wikidata_sparql_passage(binding, passages, seen_chunks)

    cache.set(cache_key, [p.__dict__ for p in passages])
    logger.debug("Cache set: %s (%d passages)", cache_key, len(passages))
    return passages
