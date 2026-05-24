"""
retrieval/bm25_retriever.py
BM25 retrieval over Wikipedia REST API results.
No LLM. No trained weights.
"""

import requests
import bm25s
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SEARCH_LIMIT = 5       # articles to fetch per query
CHUNK_SIZE = 200            # words per passage chunk
WIKI_HEADERS = {"User-Agent": "WatsonLite/1.0 (educational project; mailto:user@example.com)"}


@dataclass
class Passage:
    text: str
    source: str             # article title
    url: str
    score: float = 0.0
    rank: int = 0


def fetch_wikipedia_passages(query: str, top_k: int = WIKI_SEARCH_LIMIT) -> List[Passage]:
    """Search Wikipedia and chunk article extracts into passages."""
    # Step 1: search for relevant article titles
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": top_k,
        "format": "json",
        "utf8": 1,
    }
    try:
        resp = requests.get(WIKI_API, params=search_params, headers=WIKI_HEADERS, timeout=10)
        results = resp.json().get("query", {}).get("search", [])
    except Exception as e:
        print(f"[BM25] Wikipedia search error: {e}")
        return []

    passages = []
    for item in results:
        title = item["title"]
        # Step 2: fetch full extract for each article
        extract_params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": False,
            "explaintext": True,
            "format": "json",
        }
        try:
            eresp = requests.get(WIKI_API, params=extract_params, headers=WIKI_HEADERS, timeout=10)
            pages = eresp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                if not text:
                    continue
                # Chunk into overlapping passages
                words = text.split()
                for i in range(0, len(words), CHUNK_SIZE // 2):
                    chunk = " ".join(words[i: i + CHUNK_SIZE])
                    if len(chunk.split()) < 20:
                        continue
                    passages.append(Passage(
                        text=chunk,
                        source=title,
                        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    ))
        except Exception as e:
            print(f"[BM25] Extract error for '{title}': {e}")

    return passages


class BM25Retriever:
    def __init__(self):
        self.passages: List[Passage] = []
        self.retriever = None

    def index(self, passages: List[Passage]):
        """Build BM25 index from a list of passages."""
        self.passages = passages
        corpus = [p.text for p in passages]
        # bm25s requires corpus passed at construction to return doc text on retrieve
        tokenized = bm25s.tokenize(corpus, stopwords="en")
        self.retriever = bm25s.BM25(corpus=corpus)
        self.retriever.index(tokenized)
        print(f"[BM25] Indexed {len(passages)} passages")

    def retrieve(self, query: str, top_k: int = 10) -> List[Passage]:
        if not self.retriever or not self.passages:
            return []

        tokenized_query = bm25s.tokenize([query], stopwords="en")
        # retrieve() returns (docs, scores) where docs are corpus strings when corpus was set
        docs, scores = self.retriever.retrieve(tokenized_query, k=min(top_k, len(self.passages)))

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

    def fetch_and_retrieve(self, query: str, top_k: int = 10) -> List[Passage]:
        """Full pipeline: fetch Wikipedia → index → retrieve."""
        print(f"[BM25] Fetching Wikipedia for: '{query}'")
        passages = fetch_wikipedia_passages(query)
        if not passages:
            return []
        self.index(passages)
        return self.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    retriever = BM25Retriever()
    results = retriever.fetch_and_retrieve("Who built the Eiffel Tower?", top_k=5)
    for r in results:
        print(f"[{r.rank}] ({r.score:.2f}) {r.source}: {r.text[:100]}...")
