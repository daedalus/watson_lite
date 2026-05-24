"""
ranking/ranker.py
Reciprocal Rank Fusion (RRF) + pretrained cross-encoder reranking.
No trained weights of our own. Cross-encoder is pretrained by ms-marco team.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from sentence_transformers import CrossEncoder
from retrieval.bm25_retriever import Passage

RRF_K = 60      # standard RRF constant
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass
class RankedPassage:
    passage: Passage
    rrf_score: float = 0.0
    cross_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0


class RRFFusion:
    """Reciprocal Rank Fusion — parameter-free score fusion."""

    def fuse(self, ranked_lists: List[List[Passage]], k: int = RRF_K) -> List[Passage]:
        scores: Dict[str, float] = {}
        passage_map: Dict[str, Passage] = {}

        for ranked_list in ranked_lists:
            for rank, passage in enumerate(ranked_list, start=1):
                key = passage.text[:80]     # dedup key
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
    """Pretrained cross-encoder reranker — no training required."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        print(f"[Ranker] Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, passages: List[Passage], top_k: int = 10) -> List[RankedPassage]:
        if not passages:
            return []

        pairs = [(query, p.text) for p in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = []
        for passage, score in zip(passages, scores):
            ranked.append(RankedPassage(
                passage=passage,
                rrf_score=passage.score,
                cross_score=float(score),
            ))

        # Sort by cross-encoder score
        ranked.sort(key=lambda x: x.cross_score, reverse=True)

        for rank, rp in enumerate(ranked[:top_k], start=1):
            rp.rank = rank
            # Combined final score: normalize both to [0,1] range later
            rp.final_score = rp.cross_score

        return ranked[:top_k]


class Ranker:
    """Full ranking pipeline: RRF → cross-encoder rerank."""

    def __init__(self):
        self.rrf = RRFFusion()
        self.cross_encoder = CrossEncoderReranker()

    def rank(
        self,
        query: str,
        bm25_results: List[Passage],
        vector_results: List[Passage],
        top_k: int = 10,
    ) -> List[RankedPassage]:

        # Step 1: RRF fusion
        print(f"[Ranker] Fusing {len(bm25_results)} BM25 + {len(vector_results)} vector results")
        fused = self.rrf.fuse([bm25_results, vector_results])
        print(f"[Ranker] RRF produced {len(fused)} candidates")

        # Step 2: Cross-encoder rerank top-N for efficiency
        candidates = fused[:50]
        ranked = self.cross_encoder.rerank(query, candidates, top_k=top_k)
        print(f"[Ranker] Final top-{len(ranked)} passages ranked")

        return ranked


if __name__ == "__main__":
    from retrieval.bm25_retriever import BM25Retriever, fetch_wikipedia_passages
    from retrieval.vector_retriever import VectorRetriever

    query = "Who designed the Eiffel Tower?"
    passages = fetch_wikipedia_passages(query)

    bm25 = BM25Retriever()
    bm25.index(passages)
    bm25_results = bm25.retrieve(query, top_k=20)

    vec = VectorRetriever()
    vec.index_passages(passages)
    vec_results = vec.retrieve(query, top_k=20)

    ranker = Ranker()
    ranked = ranker.rank(query, bm25_results, vec_results, top_k=5)

    for rp in ranked:
        print(f"[{rp.rank}] cross={rp.cross_score:.3f} | {rp.passage.source}: {rp.passage.text[:120]}...")
