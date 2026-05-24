from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from watson_lite.core.cache import get_cache, is_cache_miss

if TYPE_CHECKING:
    from watson_lite.core.models import AnswerCandidate

logger = logging.getLogger(__name__)

USER_AGENT = "WatsonLite/1.0 (research project; clavijodario@gmail.com)"
_NEGATIVE_CACHE_TTL_SECONDS = 300

# Wikidata type hierarchy cache: QID -> set of ancestor QIDs
_type_cache: dict[str, set[str]] = {}


def _batch_fetch_claims(
    qids: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Fetch P31/P279 claims for a batch of QIDs via wbgetentities.

    Returns {qid: {"P31": [...], "P279": [...]}}.
    """
    if not qids:
        return {}

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "claims",
        "format": "json",
    }
    try:
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        result: dict[str, dict[str, list[str]]] = {}
        for eid, entity in data.get("entities", {}).items():
            claims = entity.get("claims", {})
            p31 = _extract_qids_from_claims(claims, "P31")
            p279 = _extract_qids_from_claims(claims, "P279")
            if p31 or p279:
                result[eid] = {"P31": p31, "P279": p279}
        return result
    except Exception as e:
        logger.warning("Batch claim fetch error: %s", e)
        return {}


def _extract_qids_from_claims(
    claims: dict[str, Any],
    pid: str,
) -> list[str]:
    """Extract QID values for a given property ID from a claims dict."""
    qids = []
    for claim in claims.get(pid, []):
        mainsnak = claim.get("mainsnak", {})
        if mainsnak.get("snaktype") != "value":
            continue
        datavalue = mainsnak.get("datavalue", {})
        val = datavalue.get("value")
        if isinstance(val, dict):
            type_qid = val.get("id", "")
            if type_qid and type_qid.startswith("Q"):
                qids.append(type_qid)
    return qids


def _fetch_type_hierarchy(qid: str, max_depth: int = 3) -> set[str]:
    """Fetch P31 (instance of) and P279 (subclass of) ancestors for a QID.

    Uses batched API calls (wbgetentities) to minimise network requests.
    Returns a set of ancestor QIDs: {direct_type, parent_type, ...}.
    """
    if qid in _type_cache:
        return _type_cache[qid]

    cache = get_cache()
    cache_key = f"tc:types:{qid}"
    cached = cache.get_or_sentinel(cache_key)
    if not is_cache_miss(cached):
        ancestors_set: set[str] = set(cached)
        _type_cache[qid] = ancestors_set
        return ancestors_set

    all_ancestors: set[str] = {qid}
    current_level = {qid}
    seen: set[str] = set()

    for _depth in range(max_depth):
        # Batch-fetch claims for all QIDs at this level (excluding cache hits).
        to_fetch = [q for q in current_level if q not in seen]
        if not to_fetch:
            break
        seen.update(to_fetch)

        fetched = _batch_fetch_claims(to_fetch)
        next_level: set[str] = set()

        for data in fetched.values():
            for pid_qids in (data.get("P31", []), data.get("P279", [])):
                for related in pid_qids:
                    if related not in all_ancestors:
                        all_ancestors.add(related)
                        next_level.add(related)

        current_level = next_level

    _type_cache[qid] = all_ancestors
    cache.set(cache_key, list(all_ancestors))
    return all_ancestors


def resolve_span_to_qid(span: str) -> str | None:
    """Resolve an answer span to a Wikidata QID via the entity search API."""
    cache = get_cache()
    cache_key = f"tc:entity:{span.lower().strip()}"
    cached = cache.get_or_sentinel(cache_key)
    if not is_cache_miss(cached):
        return str(cached) if cached else None

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": span,
        "language": "en",
        "format": "json",
    }
    try:
        resp = requests.get(
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=10
        )
        if resp.status_code != 200:
            cache.set(cache_key, None, ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
            return None
        data = resp.json()
        if data.get("search"):
            qid = str(data["search"][0]["id"])
            cache.set(cache_key, qid)
            return qid
        cache.set(cache_key, None, ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
    except Exception as e:
        logger.warning("Entity search error for '%s': %s", span, e)
        cache.set(cache_key, None, ttl_seconds=_NEGATIVE_CACHE_TTL_SECONDS)
    return None


def score_type_coercion(
    candidates: list[AnswerCandidate],
    lat_qids: list[str],
    candidate_qid: str | None = None,
) -> float:
    if not candidates or not lat_qids:
        return 0.0

    qid = candidate_qid or resolve_span_to_qid(candidates[0].span)
    if not qid:
        return 0.0

    all_types = _fetch_type_hierarchy(qid)

    for expected in lat_qids:
        if expected in all_types:
            if qid == expected:
                return 1.0
            return 0.5

    return 0.0
