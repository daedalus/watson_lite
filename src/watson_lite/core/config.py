from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FeatureConfig:
    """Runtime feature toggles and retrieval limits for WatsonLite."""

    vector_retrieval: bool = True
    query_expansion: bool = True
    graph_enrichment: bool = True
    cross_encoder_reranking: bool = True
    question_type_bonus: bool = True
    type_coercion: bool = True
    term_match: bool = True
    consistency: bool = True
    answer_merging: bool = True
    dataset_sources: tuple[str, ...] = ("wikipedia",)
    wikipedia_top_k_per_query: int = 5
    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    extraction_top_k: int = 5

    @classmethod
    def baseline(cls) -> FeatureConfig:
        return cls()

    @classmethod
    def minimal(cls) -> FeatureConfig:
        return cls(
            vector_retrieval=False,
            query_expansion=False,
            graph_enrichment=False,
            cross_encoder_reranking=False,
            question_type_bonus=False,
            type_coercion=False,
            term_match=False,
            consistency=False,
            answer_merging=False,
        )

    def with_feature(self, name: str, enabled: bool) -> FeatureConfig:
        return replace(self, **{name: enabled})  # type: ignore[arg-type]


OPTIONAL_FEATURES = (
    "vector_retrieval",
    "query_expansion",
    "graph_enrichment",
    "cross_encoder_reranking",
    "question_type_bonus",
    "type_coercion",
    "term_match",
    "consistency",
    "answer_merging",
)
