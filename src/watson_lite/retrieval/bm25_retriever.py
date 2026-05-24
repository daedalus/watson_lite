import bm25s
import requests

from watson_lite.core.cache import get_cache
from watson_lite.core.models import Passage

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SEARCH_LIMIT = 5
CHUNK_SIZE = 200
WIKI_HEADERS = {
    "User-Agent": "WatsonLite/1.0 (educational project; mailto:user@example.com)"
}


def fetch_wikipedia_passages(
    query: str, top_k: int = WIKI_SEARCH_LIMIT
) -> list[Passage]:
    cache = get_cache()
    cache_key = f"wiki:passages:{query.lower().strip()}"
    cached = cache.get(cache_key)
    if cached is not None:
        print(f"[Cache] Hit: {cache_key}")
        return [Passage(**p) for p in cached]

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": top_k,
        "format": "json",
        "utf8": 1,
    }
    try:
        resp = requests.get(
            WIKI_API, params=search_params, headers=WIKI_HEADERS, timeout=10
        )
        results = resp.json().get("query", {}).get("search", [])
    except Exception as e:
        print(f"[BM25] Wikipedia search error: {e}")
        return []

    passages = []
    for item in results:
        title = item["title"]
        extract_params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": False,
            "explaintext": True,
            "format": "json",
        }
        try:
            eresp = requests.get(
                WIKI_API, params=extract_params, headers=WIKI_HEADERS, timeout=10
            )
            pages = eresp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                if not text:
                    continue
                words = text.split()
                for i in range(0, len(words), CHUNK_SIZE // 2):
                    chunk = " ".join(words[i : i + CHUNK_SIZE])
                    if len(chunk.split()) < 20:
                        continue
                    passages.append(
                        Passage(
                            text=chunk,
                            source=title,
                            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        )
                    )
        except Exception as e:
            print(f"[BM25] Extract error for '{title}': {e}")

    cache.set(cache_key, [p.__dict__ for p in passages])
    print(f"[Cache] Set: {cache_key} ({len(passages)} passages)")
    return passages


class BM25Retriever:
    def __init__(self) -> None:
        self.passages: list[Passage] = []
        self.retriever = None

    def index(self, passages: list[Passage]) -> None:
        self.passages = passages
        corpus = [p.text for p in passages]
        tokenized = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(tokenized)
        self.retriever = retriever
        print(f"[BM25] Indexed {len(passages)} passages")

    def retrieve(self, query: str, top_k: int = 10) -> list[Passage]:
        if not self.retriever or not self.passages:
            return []
        assert self.retriever is not None  # narrowed by guard above

        tokenized_query = bm25s.tokenize([query], stopwords="en")
        docs, scores = self.retriever.retrieve(
            tokenized_query, k=min(top_k, len(self.passages))
        )

        text_to_passage = {p.text: p for p in self.passages}
        retrieved = []
        for doc_text, score in zip(docs[0], scores[0]):
            p = text_to_passage.get(doc_text)
            if p:
                p.score = float(score)
                retrieved.append(p)

        for rank, p in enumerate(retrieved):
            p.rank = rank + 1

        return retrieved

    def fetch_and_retrieve(self, query: str, top_k: int = 10) -> list[Passage]:
        print(f"[BM25] Fetching Wikipedia for: '{query}'")
        passages = fetch_wikipedia_passages(query)
        if not passages:
            return []
        self.index(passages)
        return self.retrieve(query, top_k=top_k)
