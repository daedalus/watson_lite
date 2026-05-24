import logging
import time
from typing import Any
from urllib.error import HTTPError

import requests
from SPARQLWrapper import JSON, SPARQLWrapper

from watson_lite.core.cache import get_cache, is_cache_miss
from watson_lite.core.models import EntityFact, GraphResult

logger = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "WatsonLite/1.0 (research project; clavijodario@gmail.com)"

# Static mapping of the most frequently encountered Wikidata property IDs to
# their English labels.  Unknown PIDs fall back to the raw ID string.
_PROPERTY_LABELS: dict[str, str] = {
    "P17": "country",
    "P18": "image",
    "P19": "place of birth",
    "P20": "place of death",
    "P21": "sex or gender",
    "P22": "father",
    "P25": "mother",
    "P26": "spouse",
    "P27": "country of citizenship",
    "P30": "continent",
    "P31": "instance of",
    "P36": "capital",
    "P40": "child",
    "P50": "author",
    "P57": "director",
    "P69": "educated at",
    "P84": "architect",
    "P101": "field of work",
    "P106": "occupation",
    "P108": "employer",
    "P112": "founded by",
    "P131": "located in",
    "P150": "contains",
    "P159": "headquarters",
    "P166": "award received",
    "P169": "chief executive officer",
    "P170": "creator",
    "P175": "performer",
    "P276": "location",
    "P279": "subclass of",
    "P463": "member of",
    "P495": "country of origin",
    "P527": "has part",
    "P569": "date of birth",
    "P570": "date of death",
    "P571": "inception",
    "P576": "dissolved",
    "P577": "publication date",
    "P625": "coordinate location",
    "P856": "official website",
    "P1082": "population",
    "P2044": "elevation above sea level",
}


class WikidataGraph:
    def __init__(self) -> None:
        self.sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
        self.sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
        self.sparql.setReturnFormat(JSON)

    def _run_query(self, query: str, retries: int = 3) -> list[dict[str, object]]:
        for attempt in range(retries):
            try:
                self.sparql.setQuery(query)
                data: Any = self.sparql.query().convert()
                return list(data["results"]["bindings"])
            except HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    # Exponential back-off: 2s, 4s, 8s, …
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Rate limited by SPARQL endpoint, retrying in %ss "
                        "(attempt %d/%d)",
                        wait,
                        attempt + 2,
                        retries,
                    )
                    time.sleep(wait)
                    continue
                logger.warning("SPARQL HTTP error: %s", e)
                return []
            except Exception as e:
                logger.warning("SPARQL query error: %s", e)
                return []
        return []

    def find_entity_id(self, entity_name: str) -> str | None:
        cache = get_cache()
        cache_key = f"wd:entity:{entity_name.lower().strip()}"
        cached = cache.get_or_sentinel(cache_key)
        if not is_cache_miss(cached):
            return str(cached) if cached is not None else None

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": entity_name,
            "language": "en",
            "format": "json",
        }
        try:
            resp = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=10
            )
            if resp.status_code == 429:
                logger.warning("Rate limited on entity search, falling back to SPARQL")
                qid = self._find_entity_id_sparql(entity_name)
                if qid:
                    cache.set(cache_key, qid)
                return qid
            data = resp.json()
            if data.get("search"):
                qid = str(data["search"][0]["id"])
                cache.set(cache_key, qid)
                return qid
        except Exception as e:
            logger.warning("Entity search error: %s", e)
        return None

    def _find_entity_id_sparql(self, entity_name: str) -> str | None:
        # Escape double-quotes to prevent SPARQL injection.
        safe_name = entity_name.replace("\\", "\\\\").replace('"', '\\"')
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?item WHERE {{
          ?item rdfs:label "{safe_name}"@en .
        }}
        LIMIT 1
        """
        rows = self._run_query(query)
        if rows:
            item: Any = rows[0]["item"]
            uri: str = item["value"]
            return uri.split("/")[-1]
        return None

    def _resolve_qid_labels(self, qids: set[str]) -> dict[str, str]:
        """Batch-resolve QIDs to English labels via the Wikidata API.

        Returns a dict mapping QID → label.  QIDs that fail to resolve are
        omitted so callers fall back to the raw QID.
        """
        if not qids:
            return {}
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": "|".join(qids),
            "props": "labels",
            "languages": "en",
            "format": "json",
        }
        try:
            resp = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15
            )
            if resp.status_code != 200:
                return {}
            data = resp.json()
            entities = data.get("entities", {})
            return {
                eid: info["labels"]["en"]["value"]
                for eid, info in entities.items()
                if "labels" in info and "en" in info["labels"]
            }
        except Exception:
            logger.warning("Failed to resolve QID labels", exc_info=True)
            return {}

    def _parse_claims_to_facts(
        self, qid: str, claims: dict, max_facts: int
    ) -> tuple[list[EntityFact], set[str]]:
        facts: list[EntityFact] = []
        seen: set[str] = set()
        qid_values: set[str] = set()
        for pid, claim_list in claims.items():
            for claim in claim_list[:3]:
                mainsnak = claim.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue
                datavalue = mainsnak.get("datavalue", {})
                val = datavalue.get("value")
                is_qid = isinstance(val, dict)
                if is_qid:
                    raw_val = str(val.get("id", ""))
                else:
                    raw_val = str(val)
                if raw_val in seen or len(facts) >= max_facts:
                    continue
                seen.add(raw_val)
                if is_qid and raw_val.startswith("Q"):
                    qid_values.add(raw_val)
                facts.append(
                    EntityFact(
                        entity=qid,
                        property_label=_PROPERTY_LABELS.get(pid, pid),
                        value=raw_val,
                        value_type=datavalue.get("type", "literal"),
                    )
                )
        return facts, qid_values

    def get_entity_facts(self, qid: str, max_facts: int = 15) -> list[EntityFact]:
        cache = get_cache()
        cache_key = f"wd:facts:{qid}"
        cached = cache.get_or_sentinel(cache_key)
        if not is_cache_miss(cached):
            return [EntityFact(**f) for f in cached]

        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if resp.status_code != 200:
                logger.warning("EntityData request failed: HTTP %s", resp.status_code)
                return []
            data = resp.json()
            entity = data.get("entities", {}).get(qid, {})
            facts, qid_values = self._parse_claims_to_facts(
                qid, entity.get("claims", {}), max_facts
            )
            if qid_values:
                labels = self._resolve_qid_labels(qid_values)
                for f in facts:
                    if f.value in labels:
                        f.value = labels[f.value]
            cache.set(cache_key, [f.__dict__ for f in facts])
            return facts
        except Exception as e:
            logger.warning("EntityData error: %s", e)
            return []

    def get_related_entities(self, qid: str, max_related: int = 10) -> list[str]:
        """Return Wikidata entity IDs referenced in the entity's own claims.

        Extracts QIDs from the already-cached facts for *qid* rather than
        making additional network requests.  Returns an empty list when the
        facts for *qid* are not yet in the cache.
        """
        cache = get_cache()
        cache_key = f"wd:facts:{qid}"
        cached = cache.get_or_sentinel(cache_key)
        if is_cache_miss(cached):
            return []
        facts: list[EntityFact] = [EntityFact(**f) for f in cached]
        seen: set[str] = set()
        related: list[str] = []
        for fact in facts:
            val = fact.value
            if not isinstance(val, str):
                continue
            is_entity = fact.value_type == "wikibase-entityid" or (
                val.startswith("Q") and val[1:].isdigit()
            )
            if is_entity and val not in seen:
                seen.add(val)
                related.append(val)
                if len(related) >= max_related:
                    break
        return related

    @staticmethod
    def _clean_entity_name(name: str) -> str:
        cleaned = name.strip()
        for article in ("the ", "a ", "an "):
            if cleaned.lower().startswith(article):
                cleaned = cleaned[len(article) :]
                break
        return cleaned.strip()

    def enrich(self, entity_name: str) -> GraphResult:
        cleaned = self._clean_entity_name(entity_name)
        if cleaned:
            logger.debug("Enriching entity: '%s' -> '%s'", entity_name, cleaned)
        else:
            cleaned = entity_name
            logger.debug("Enriching entity: '%s'", entity_name)
        qid = self.find_entity_id(cleaned)
        result = GraphResult(entity_name=entity_name, wikidata_id=qid)

        if qid:
            result.facts = self.get_entity_facts(qid)
            result.related_entities = self.get_related_entities(qid)
            logger.debug(
                "Found %d facts, %d related entities for %s",
                len(result.facts),
                len(result.related_entities),
                qid,
            )
        else:
            logger.debug("No Wikidata ID found for '%s'", cleaned)

        return result

    def enrich_all(self, entity_names: list[str]) -> list[GraphResult]:
        return [self.enrich(name) for name in entity_names]
