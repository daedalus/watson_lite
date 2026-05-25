import copy
import logging
from typing import Any

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - exercised via lazy init tests
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - exercised via lazy init tests
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

from watson_lite.core.models import Passage

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"


class VectorRetriever:
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        if SentenceTransformer is None or faiss is None:
            missing = []
            if SentenceTransformer is None:
                missing.append("sentence-transformers")
            if faiss is None:
                missing.append("faiss-cpu")
            raise ImportError(
                "Vector retrieval dependencies are missing: "
                f"{', '.join(missing)}. Install watson-lite with the "
                "'vector' or 'full' extra."
            ) from (_SENTENCE_TRANSFORMERS_IMPORT_ERROR or _FAISS_IMPORT_ERROR)
        logger.debug("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index: Any = None  # faiss.IndexFlatIP once built
        self.passages: list[Passage] = []
        self.dim = self.model.get_sentence_embedding_dimension()

    def index_passages(self, passages: list[Passage]) -> None:
        self.passages = passages
        texts = [p.text for p in passages]

        logger.debug("Encoding %d passages...", len(texts))
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        self.index = index
        logger.debug("FAISS index built: %d vectors", index.ntotal)

    def retrieve(self, query: str, top_k: int = 10) -> list[Passage]:
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
            # Copy so that setting score/rank does not mutate the shared
            # Passage stored in self.passages.
            p = copy.copy(self.passages[idx])
            p.score = float(score)
            p.rank = rank + 1
            retrieved.append(p)

        return retrieved
