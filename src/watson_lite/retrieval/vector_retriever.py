import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from watson_lite.core.models import Passage

EMBED_MODEL = "all-MiniLM-L6-v2"


class VectorRetriever:
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        print(f"[Vector] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.passages: list[Passage] = []
        self.dim = self.model.get_sentence_embedding_dimension()

    def index_passages(self, passages: list[Passage]) -> None:
        self.passages = passages
        texts = [p.text for p in passages]

        print(f"[Vector] Encoding {len(texts)} passages...")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        self.index = index
        print(f"[Vector] FAISS index built: {index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 10) -> list[Passage]:
        if self.index is None or not self.passages:
            return []
        assert self.index is not None

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
