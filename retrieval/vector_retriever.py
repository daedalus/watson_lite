"""
retrieval/vector_retriever.py
Dense vector retrieval using sentence-transformers + FAISS.
No LLM. Pretrained embedding model only (inference, no training).
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List
from retrieval.bm25_retriever import Passage


EMBED_MODEL = "all-MiniLM-L6-v2"   # Fast, CPU-friendly, 384-dim


class VectorRetriever:
    def __init__(self, model_name: str = EMBED_MODEL):
        print(f"[Vector] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.passages: List[Passage] = []
        self.dim = self.model.get_sentence_embedding_dimension()

    def index_passages(self, passages: List[Passage]):
        """Encode passages and build FAISS index."""
        self.passages = passages
        texts = [p.text for p in passages]

        print(f"[Vector] Encoding {len(texts)} passages...")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        print(f"[Vector] FAISS index built: {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 10) -> List[Passage]:
        if self.index is None or not self.passages:
            return []

        query_vec = self.model.encode([query], show_progress_bar=False)
        query_vec = np.array(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, min(top_k, len(self.passages)))

        retrieved = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            p = self.passages[idx]
            p.score = float(score)
            p.rank = rank + 1
            retrieved.append(p)

        return retrieved

    def save_index(self, path: str):
        if self.index:
            faiss.write_index(self.index, path)
            print(f"[Vector] Index saved to {path}")

    def load_index(self, path: str, passages: List[Passage]):
        self.index = faiss.read_index(path)
        self.passages = passages
        print(f"[Vector] Index loaded from {path}")


if __name__ == "__main__":
    from retrieval.bm25_retriever import fetch_wikipedia_passages

    passages = fetch_wikipedia_passages("Eiffel Tower construction history")
    retriever = VectorRetriever()
    retriever.index_passages(passages)
    results = retriever.retrieve("Who designed the Eiffel Tower?", top_k=5)
    for r in results:
        print(f"[{r.rank}] ({r.score:.3f}) {r.source}: {r.text[:100]}...")
