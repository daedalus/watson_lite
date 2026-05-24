import hashlib
import logging

from sentence_transformers import CrossEncoder

from watson_lite.core.models import Passage, RankedPassage

logger = logging.getLogger(__name__)

RRF_K = 60
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class RRFFusion:
    def fuse(self, ranked_lists: list[list[Passage]], k: int = RRF_K) -> list[Passage]:
        scores: dict[str, float] = {}
        passage_map: dict[str, Passage] = {}

        for ranked_list in ranked_lists:
            for rank, passage in enumerate(ranked_list, start=1):
                # Use a full-text hash to avoid collisions between passages
                # whose first 80 characters happen to be identical.
                key = hashlib.md5(passage.text.encode()).hexdigest()
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                passage_map[key] = passage

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (key, score) in enumerate(fused, start=1):
            p = passage_map[key]
            p.score = score
            p.rank = rank
            results.append(p)

        return results


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        logger.debug("Loading cross-encoder: %s", model_name)
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self, query: str, passages: list[Passage], top_k: int = 10
    ) -> list[RankedPassage]:
        if not passages:
            return []

        pairs = [(query, p.text) for p in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = []
        for passage, score in zip(passages, scores):
            ranked.append(
                RankedPassage(
                    passage=passage,
                    rrf_score=passage.score,
                    cross_score=float(score),
                )
            )

        ranked.sort(key=lambda x: x.cross_score, reverse=True)

        for rank, rp in enumerate(ranked[:top_k], start=1):
            rp.rank = rank
            rp.final_score = rp.cross_score

        return ranked[:top_k]


class Ranker:
    def __init__(self) -> None:
        self.rrf = RRFFusion()
        self.cross_encoder = CrossEncoderReranker()

    def rank(
        self,
        query: str,
        bm25_results: list[Passage],
        vector_results: list[Passage],
        top_k: int = 10,
    ) -> list[RankedPassage]:

        logger.debug(
            "Fusing %d BM25 + %d vector results",
            len(bm25_results),
            len(vector_results),
        )
        fused = self.rrf.fuse([bm25_results, vector_results])
        logger.debug("RRF produced %d candidates", len(fused))

        candidates = fused[:50]
        ranked = self.cross_encoder.rerank(query, candidates, top_k=top_k)
        logger.debug("Final top-%d passages ranked", len(ranked))

        return ranked
