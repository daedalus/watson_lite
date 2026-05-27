from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FeatureConfig:
    """Runtime feature toggles and retrieval limits for WatsonLite."""

    vector_retrieval: bool = True
    query_expansion: bool = True
    query_context_augmentation: bool = True
    graph_enrichment: bool = True
    cross_encoder_reranking: bool = True
    question_type_bonus: bool = True
    type_coercion: bool = True
    term_match: bool = True
    consistency: bool = True
    entailment: bool = True
    answer_merging: bool = True
    multi_hypothesis: bool = True
    per_candidate_retrieval: bool = True
    bidirectional_validation: bool = True
    iterative_retrieval: bool = True
    semantic_nlp: bool = False
    index_dir: str | None = None
    dataset_sources: tuple[str, ...] = ("wikipedia",)
    elasticsearch_url: str | None = None
    elasticsearch_index: str | None = None
    huggingface_dataset: str | None = None
    huggingface_config: str | None = None
    huggingface_split: str | None = None
    huggingface_token: str | None = None
    offline_dataset_dir: str | None = None
    wikipedia_top_k_per_query: int = 5
    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    extraction_top_k: int = 5
    max_retrieval_passes: int = 2
    iterative_retrieval_threshold: float = 0.3
    confidence_threshold: float | None = None
    spacy_model: str | None = None
    embed_model: str | None = None
    cross_encoder_model: str | None = None
    nli_model: str | None = None

    @classmethod
    def baseline(cls) -> FeatureConfig:
        return cls()

    @classmethod
    def minimal(cls) -> FeatureConfig:
        return cls(
            vector_retrieval=False,
            query_expansion=False,
            query_context_augmentation=False,
            graph_enrichment=False,
            cross_encoder_reranking=False,
            question_type_bonus=False,
            type_coercion=False,
            term_match=False,
            consistency=False,
            entailment=False,
            answer_merging=False,
            multi_hypothesis=False,
            per_candidate_retrieval=False,
            bidirectional_validation=False,
            iterative_retrieval=False,
        )

    def with_feature(self, name: str, enabled: bool) -> FeatureConfig:
        return replace(self, **{name: enabled})  # type: ignore[arg-type]


OPTIONAL_FEATURES = (
    "vector_retrieval",
    "query_expansion",
    "query_context_augmentation",
    "graph_enrichment",
    "cross_encoder_reranking",
    "question_type_bonus",
    "type_coercion",
    "term_match",
    "consistency",
    "entailment",
    "answer_merging",
    "multi_hypothesis",
    "per_candidate_retrieval",
    "bidirectional_validation",
    "iterative_retrieval",
    "semantic_nlp",
)
