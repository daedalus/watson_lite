import time
from urllib.error import HTTPError

import requests
from SPARQLWrapper import JSON, SPARQLWrapper

from watson_lite.core.cache import get_cache
from watson_lite.core.models import EntityFact, GraphResult

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "WatsonLite/1.0 (research project; python)"


class WikidataGraph:
    def __init__(self) -> None:
        self.sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
        self.sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
        self.sparql.setReturnFormat(JSON)

    def _run_query(self, query: str, retries: int = 3) -> list[dict[str, object]]:
        for attempt in range(retries):
            try:
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                return results["results"]["bindings"]
            except HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    wait = 30 * (attempt + 1)
                    print(
                        f"[Graph] Rate limited, retrying in {wait}s (attempt {attempt + 2}/{retries})"
                    )
                    time.sleep(wait)
                    continue
                print(f"[Graph] SPARQL error: {e}")
                return []
            except Exception as e:
                print(f"[Graph] SPARQL error: {e}")
                return []
        return []

    def find_entity_id(self, entity_name: str) -> str | None:
        cache = get_cache()
        cache_key = f"wd:entity:{entity_name.lower().strip()}"
        cached = cache.get(cache_key)
        if cached is not None:
            return str(cached)

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
                print("[Graph] Rate limited on entity search, falling back to SPARQL")
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
            print(f"[Graph] Entity search error: {e}")
        return None

    def _find_entity_id_sparql(self, entity_name: str) -> str | None:
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?item WHERE {{
          ?item rdfs:label "{entity_name}"@en .
        }}
        LIMIT 1
        """
        rows = self._run_query(query)
        if rows:
            uri = rows[0]["item"]["value"]
            return uri.split("/")[-1]
        return None

    def get_entity_facts(self, qid: str, max_facts: int = 15) -> list[EntityFact]:
        cache = get_cache()
        cache_key = f"wd:facts:{qid}"
        cached = cache.get(cache_key)
        if cached is not None:
            return [EntityFact(**f) for f in cached]

        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if resp.status_code != 200:
                print(f"[Graph] EntityData error: HTTP {resp.status_code}")
                return []
            data = resp.json()
            entity = data.get("entities", {}).get(qid, {})
            claims = entity.get("claims", {})

            facts: list[EntityFact] = []
            seen: set[str] = set()
            for pid, claim_list in claims.items():
                for claim in claim_list[:3]:
                    mainsnak = claim.get("mainsnak", {})
                    if mainsnak.get("snaktype") != "value":
                        continue
                    datavalue = mainsnak.get("datavalue", {})
                    val = datavalue.get("value")
                    if isinstance(val, dict):
                        val = val.get("id", str(val))
                    val = str(val)
                    if val in seen or len(facts) >= max_facts:
                        continue
                    seen.add(val)
                    facts.append(
                        EntityFact(
                            entity=qid,
                            property_label=pid,
                            value=val,
                            value_type=datavalue.get("type", "literal"),
                        )
                    )
            if facts:
                cache.set(cache_key, [f.__dict__ for f in facts])
            return facts
        except Exception as e:
            print(f"[Graph] EntityData error: {e}")
            return []

    def get_related_entities(self, _qid: str, _max_related: int = 10) -> list[str]:
        return []

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
            print(f"[Graph] Enriching entity: '{entity_name}' -> '{cleaned}'")
        else:
            cleaned = entity_name
            print(f"[Graph] Enriching entity: '{entity_name}'")
        qid = self.find_entity_id(cleaned)
        result = GraphResult(entity_name=entity_name, wikidata_id=qid)

        if qid:
            result.facts = self.get_entity_facts(qid)
            result.related_entities = self.get_related_entities(qid)
            print(
                f"[Graph] Found {len(result.facts)} facts, {len(result.related_entities)} related entities for {qid}"
            )
        else:
            print(f"[Graph] No Wikidata ID found for '{cleaned}'")

        return result

    def enrich_all(self, entity_names: list[str]) -> list[GraphResult]:
        return [self.enrich(name) for name in entity_names]
