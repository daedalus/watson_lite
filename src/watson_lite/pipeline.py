import hashlib
import logging
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import cast

import langdetect

from watson_lite.core.cache import CacheMetrics, get_cache_metrics_snapshot
from watson_lite.core.config import FeatureConfig
from watson_lite.core.extractor import ConfidenceScorer, ExtractiveReader
from watson_lite.core.fallbacks import is_fallback_answer_text
from watson_lite.core.models import (
    AnswerCandidate,
    AnswerDiagnostics,
    FinalAnswer,
    GraphResult,
    ParsedQuestion,
    Passage,
    RankedPassage,
)
from watson_lite.core.nlp import NLPProcessor
from watson_lite.graph.wikidata import WikidataGraph
from watson_lite.ranking.ranker import Ranker
from watson_lite.retrieval.bm25_retriever import (
    BM25Retriever,
    fetch_wikipedia_page_by_title,
)
from watson_lite.retrieval.dataset_plugins import build_dataset_plugin_registry
from watson_lite.retrieval.dataset_query_engine import DatasetQueryEngine
from watson_lite.retrieval.query_formulation import generate_search_queries
from watson_lite.retrieval.vector_retriever import EMBED_MODEL, VectorRetriever
from watson_lite.scoring.double_check import bidirectional_score
from watson_lite.scoring.entailment import configure_entailment_model

logger = logging.getLogger(__name__)
_NON_WORD = re.compile(r"\W+")

_ENGLISH_EMBED_MODEL = "all-MiniLM-L6-v2"
_ENGLISH_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_ENGLISH_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_MULTILINGUAL_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_MULTILINGUAL_CE_MODEL = "cross-encoder/stsb-distilroberta-base"
_MULTILINGUAL_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-mnli"
_ENGLISH_EXTRACTIVE_MODEL = "deepset/roberta-base-squad2"
_MULTILINGUAL_EXTRACTIVE_MODEL = "deepset/xlm-roberta-base-squad2"


def _passage_content_key(p: Passage) -> str:
    """Return a stable content hash for a single passage."""
    parts = f"{p.source}:{p.url}:{p.text}"
    return hashlib.sha256(parts.encode()).hexdigest()


def _passage_dedup_key(passage: Passage) -> str:
    normalized = _NON_WORD.sub(" ", passage.text.lower())
    return " ".join(normalized.split())


class WatsonLite:
    def __init__(self, config: FeatureConfig | None = None, device: int = -1) -> None:
        logger.info("WatsonLite — Initializing pipeline (device=%d)", device)
        self.config = config or FeatureConfig.baseline()
        self.device = device
        self.bm25 = BM25Retriever()
        self.nlp: NLPProcessor | None = None
        self.vector: VectorRetriever | None = None
        self.dataset_plugins = build_dataset_plugin_registry(self.config)
        self.dataset_query_engine = DatasetQueryEngine(
            providers=self.dataset_plugins.provider_tuple(),
            enabled_datasets=self.config.dataset_sources,
        )
        self.graph: WikidataGraph | None = None
        self.ranker: Ranker | None = None
        self.reader: ExtractiveReader | None = None
        self.scorer = ConfidenceScorer(
            confidence_threshold=self.config.confidence_threshold
        )
        if self.config.nli_model is not None:
            configure_entailment_model(self.config.nli_model)
        # Per-passage content-addressable cache: key = _passage_content_key
        self._passage_cache: dict[str, Passage] = {}
        self._index_loaded = False
        self._nlp_cache: dict[str, NLPProcessor] = {}
        self._vector_cache: dict[str, VectorRetriever] = {}
        self._ranker_cache: dict[str, Ranker] = {}
        self._reader_cache: dict[str, ExtractiveReader] = {}
        self._current_embed_model: str = _MULTILINGUAL_EMBED_MODEL
        self._current_ce_model: str = _MULTILINGUAL_CE_MODEL
        self._current_nli_model: str = _ENGLISH_NLI_MODEL
        self._current_extractive_model: str = _ENGLISH_EXTRACTIVE_MODEL
        self._load_prebuilt_index()
        logger.info("Core components loaded. Heavy models will load lazily.")

    def _load_prebuilt_index(self) -> None:
        index_dir = self.config.index_dir
        if index_dir is None:
            return
        bm25_path = os.path.join(index_dir, "bm25")
        vector_path = os.path.join(index_dir, "vector")
        if os.path.isdir(bm25_path):
            self.bm25 = BM25Retriever.load(bm25_path)
            self._passage_cache = {
                _passage_content_key(p): p for p in self.bm25.passages
            }
            logger.info("Loaded pre-built BM25 index from %s", bm25_path)
        if os.path.isdir(vector_path) and self.config.vector_retrieval:
            self.vector = None
            try:
                embed_model = self.config.embed_model or self._current_embed_model
                loaded = VectorRetriever.load(vector_path, model_name=embed_model)
                self._vector_cache[embed_model] = loaded
                self.vector = loaded
            except ImportError:
                logger.warning("Vector dependencies missing; skipping FAISS index load")
        self._index_loaded = True

    def _select_models_for_language(self, language: str) -> None:
        """Set embedding, cross-encoder, and NLI models based on detected language."""
        if self.config.embed_model is None:
            self._current_embed_model = (
                _ENGLISH_EMBED_MODEL if language == "en" else _MULTILINGUAL_EMBED_MODEL
            )
        if self.config.cross_encoder_model is None:
            self._current_ce_model = (
                _ENGLISH_CE_MODEL if language == "en" else _MULTILINGUAL_CE_MODEL
            )
        if self.config.nli_model is None:
            self._current_nli_model = (
                _ENGLISH_NLI_MODEL if language == "en" else _MULTILINGUAL_NLI_MODEL
            )
        else:
            self._current_nli_model = self.config.nli_model
        self._current_extractive_model = (
            _ENGLISH_EXTRACTIVE_MODEL
            if language == "en"
            else _MULTILINGUAL_EXTRACTIVE_MODEL
        )

    def _get_nlp(self, language: str = "en") -> NLPProcessor:
        if language not in self._nlp_cache:
            self._nlp_cache[language] = NLPProcessor(
                model=self.config.spacy_model,
                language=language,
                semantic_nlp=self.config.semantic_nlp,
            )
        return self._nlp_cache[language]

    def _get_vector(self) -> VectorRetriever | None:
        if not self.config.vector_retrieval:
            return None
        model = self.config.embed_model or self._current_embed_model
        if model not in self._vector_cache:
            self._vector_cache[model] = VectorRetriever(model_name=model)
        self.vector = self._vector_cache[model]
        return self.vector

    def _get_graph(self) -> WikidataGraph:
        if self.graph is None:
            self.graph = WikidataGraph(
                sparql_endpoint=self.config.wikidata_sparql_endpoint,
            )
        return self.graph

    def _get_ranker(self) -> Ranker:
        key = self.config.cross_encoder_model or self._current_ce_model
        if key not in self._ranker_cache:
            self._ranker_cache[key] = Ranker(
                enable_cross_encoder=self.config.cross_encoder_reranking,
                cross_encoder_model=key,
            )
        self.ranker = self._ranker_cache[key]
        return self.ranker

    def _get_reader(self) -> ExtractiveReader:
        if self.reader is not None:
            return self.reader
        model = self._current_extractive_model
        if model not in self._reader_cache:
            self._reader_cache[model] = ExtractiveReader(
                model_name=model, device=self.device
            )
        return self._reader_cache[model]

    def _retrieve_parallel(  # pylint: disable=too-many-arguments
        self,
        question: str,
        passages: list[Passage],
        needs_reindex: bool,
        *,
        vector_retriever: VectorRetriever | None,
        vector_enabled: bool,
        top_k: int,
    ) -> tuple[list[Passage], list[Passage]]:
        """Run BM25 and vector retrieval in parallel, re-indexing only when needed."""

        def _bm25_work() -> list[Passage]:
            if needs_reindex:
                self.bm25.index(passages)
            return self.bm25.retrieve(question, top_k=top_k)

        if not vector_enabled or vector_retriever is None:
            return _bm25_work(), []

        def _vector_work() -> list[Passage]:
            if needs_reindex:
                vector_retriever.index_passages(passages)
            return cast(
                "list[Passage]",
                vector_retriever.retrieve(question, top_k=top_k),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(_bm25_work)
            vector_future = executor.submit(_vector_work)
            return bm25_future.result(), vector_future.result()

    @staticmethod
    def _log_step(verbose: bool, num: int, message: str) -> None:
        if verbose:
            logger.info("[%d/6] %s", num, message)

    @staticmethod
    def _log_detail(verbose: bool, fmt: str, *args: object) -> None:
        if verbose:
            logger.info("      " + fmt, *args)

    def _log_graph_results(
        self, verbose: bool, graph_results: list[GraphResult]
    ) -> None:
        if verbose:
            for gr in graph_results:
                logger.info("      %s: %d facts", gr.entity_name, len(gr.facts))

    @staticmethod
    def _cache_metrics_delta(
        before: CacheMetrics,
        after: CacheMetrics,
    ) -> tuple[int, int, dict[str, int], dict[str, int]]:
        before_hits = before["hits"]
        after_hits = after["hits"]
        before_misses = before["misses"]
        after_misses = after["misses"]

        before_hits_ns = before["hits_by_namespace"]
        after_hits_ns = after["hits_by_namespace"]
        before_misses_ns = before["misses_by_namespace"]
        after_misses_ns = after["misses_by_namespace"]

        hits_by_ns: dict[str, int] = {}
        misses_by_ns: dict[str, int] = {}

        for key, value in after_hits_ns.items():
            hits_by_ns[str(key)] = int(value) - int(before_hits_ns.get(key, 0))
        for key, value in after_misses_ns.items():
            misses_by_ns[str(key)] = int(value) - int(before_misses_ns.get(key, 0))

        return (
            max(0, after_hits - before_hits),
            max(0, after_misses - before_misses),
            {k: v for k, v in hits_by_ns.items() if v > 0},
            {k: v for k, v in misses_by_ns.items() if v > 0},
        )

    @staticmethod
    def _is_fallback_answer(answer: FinalAnswer) -> bool:
        return cast("bool", is_fallback_answer_text(answer.answer))

    @staticmethod
    @staticmethod
    def _dedupe_passages(passages: list[Passage]) -> list[Passage]:
        seen_texts: set[str] = set()
        deduped: list[Passage] = []
        for passage in passages:
            dedup_key = _passage_dedup_key(passage)
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)
            deduped.append(passage)
        return deduped

    def _needs_reindex(self, passages: list[Passage]) -> bool:
        new_keys = {_passage_content_key(p) for p in passages}
        existing_keys = set(self._passage_cache.keys())
        if new_keys == existing_keys:
            return False
        self._passage_cache = {_passage_content_key(p): p for p in passages}
        return True

    def _run_retrieval(
        self,
        question: str,
        parsed: ParsedQuestion,
        entity_names: list[str],
        stage_t0: float,
        verbose: bool,
    ) -> tuple[int, list[Passage], list[Passage], float] | None:
        if self._index_loaded and not self.config.query_expansion:
            queries = [question]
        else:
            queries = self._generate_queries(parsed)

        if self._index_loaded:
            self._log_detail(
                verbose,
                "Pre-built index available (%d passages); fetching online for completeness",
                len(self._passage_cache),
            )

        self._log_detail(verbose, "Search queries: %s", queries)

        all_passages: list[Passage] = []
        for query in queries:
            all_passages.extend(
                self.dataset_query_engine.query(
                    query, top_k=self.config.wikipedia_top_k_per_query
                )
            )

        if entity_names:
            self._log_detail(verbose, "Entity direct page fetch: %s", entity_names)
            for entity_text in entity_names:
                all_passages.extend(fetch_wikipedia_page_by_title(entity_text))

        passages = self._dedupe_passages(all_passages)
        if not passages:
            return None

        needs_reindex = self._needs_reindex(passages)
        bm25_results, vector_results = self._retrieve_parallel(
            question,
            passages,
            needs_reindex,
            vector_retriever=self._get_vector(),
            vector_enabled=self.config.vector_retrieval,
            top_k=self.config.retrieval_top_k,
        )
        elapsed = round(time.perf_counter() - stage_t0, 4)
        return len(passages), bm25_results, vector_results, elapsed

    def _extract_candidates_for_sub_question(
        self,
        sub_question: str,
        ranked: list[RankedPassage],
    ) -> tuple[list[AnswerCandidate], int]:
        extraction_result = self._get_reader().extract(
            sub_question,
            ranked,
            top_k=self.config.extraction_top_k,
            return_stats=True,
        )
        if isinstance(extraction_result, tuple):
            return extraction_result
        return extraction_result, 0

    def _collect_candidates(
        self,
        parsed: ParsedQuestion,
        ranked: list[RankedPassage],
        *,
        verbose: bool,
    ) -> tuple[list[AnswerCandidate], int]:
        all_candidates: list[AnswerCandidate] = []
        extraction_errors = 0

        if self.config.multi_hypothesis and len(parsed.sub_questions) > 1:
            max_workers = min(len(parsed.sub_questions), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: list[Future[tuple[list[AnswerCandidate], int]]] = [
                    executor.submit(
                        self._extract_candidates_for_sub_question,
                        sub_question,
                        ranked,
                    )
                    for sub_question in parsed.sub_questions
                ]
                for future in futures:
                    candidates, errors = future.result()
                    all_candidates.extend(candidates)
                    extraction_errors += errors
        else:
            for sub_question in parsed.sub_questions:
                candidates, errors = self._extract_candidates_for_sub_question(
                    sub_question, ranked
                )
                all_candidates.extend(candidates)
                extraction_errors += errors

        if self.config.multi_hypothesis:
            for entity in parsed.entities:
                entity_text = str(entity["text"]).strip()
                if not entity_text:
                    continue
                all_candidates.append(
                    AnswerCandidate(
                        span=entity_text,
                        source="title_match",
                        url="",
                        passage=f"Entity mentioned in question: {entity_text}",
                        extraction_score=0.01,
                        rank=99,
                    )
                )

        all_candidates.sort(key=lambda c: c.extraction_score, reverse=True)
        self._log_detail(verbose, "Candidates collected: %d", len(all_candidates))
        return all_candidates, extraction_errors

    def _per_candidate_reretrieval(
        self,
        parsed: ParsedQuestion,
        candidates: list[AnswerCandidate],
        *,
        verbose: bool,
    ) -> float:
        if not self.config.per_candidate_retrieval or not candidates:
            return 0.0

        self._log_detail(verbose, "Per-candidate re-retrieval")
        query_suffix = " ".join(parsed.keywords[:3]).strip()
        seen_spans: set[str] = set()
        top_spans: list[str] = []
        for candidate in candidates:
            key = candidate.span.lower().strip()
            if not key or key in seen_spans:
                continue
            seen_spans.add(key)
            top_spans.append(candidate.span)
            if len(top_spans) == 3:
                break

        candidates_by_span: dict[str, list[AnswerCandidate]] = {}
        for candidate in candidates:
            candidates_by_span.setdefault(candidate.span.lower().strip(), []).append(
                candidate
            )

        stage_t0 = time.perf_counter()
        for span in top_spans:
            query = f"{span} {query_suffix}".strip()
            for passage in self.dataset_query_engine.query(query, top_k=2):
                passage_text = passage.text.lower()
                for span_key, matching_candidates in candidates_by_span.items():
                    if span_key and span_key in passage_text:
                        for matching_candidate in matching_candidates:
                            matching_candidate.doc_frequency += 1
        return round(time.perf_counter() - stage_t0, 4)

    def _score_answer(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        question: str,
        parsed: ParsedQuestion,
        ranked: list[RankedPassage],
        candidates: list[AnswerCandidate],
        graph_results: list[GraphResult],
        *,
        bidirectional_signal: float,
    ) -> FinalAnswer:
        return self.scorer.score(
            candidates,
            graph_results,
            parsed.question_word_type,
            lat_qids=parsed.lat_qids,
            question=question,
            ranked_passages=ranked,
            enable_question_type_bonus=self.config.question_type_bonus,
            enable_type_coercion=self.config.type_coercion,
            enable_term_match=self.config.term_match,
            enable_consistency=self.config.consistency,
            enable_entailment=self.config.entailment,
            enable_answer_merging=self.config.answer_merging,
            bidirectional_signal=bidirectional_signal,
        )

    def answer(self, question: str, verbose: bool = True) -> FinalAnswer:
        if not question:
            raise ValueError("question must not be empty")

        t0 = time.perf_counter()
        stage_latencies: dict[str, float] = {}
        cache_before = get_cache_metrics_snapshot()

        try:
            language = langdetect.detect(question)
        except Exception:
            language = "en"

        self._select_models_for_language(language)

        self._log_step(verbose, 1, "NLP preprocessing...")
        stage_t0 = time.perf_counter()
        parsed = self._get_nlp(language).process(
            question, semantic_nlp=self.config.semantic_nlp
        )
        stage_latencies["nlp"] = round(time.perf_counter() - stage_t0, 4)
        self._log_detail(verbose, "Type: %s", parsed.question_type)
        self._log_detail(verbose, "Entities: %s", [e["text"] for e in parsed.entities])
        self._log_detail(verbose, "Sub-questions: %s", parsed.sub_questions)

        self._log_step(verbose, 2, "Parallel retrieval (BM25 + Vector)...")
        stage_t0 = time.perf_counter()

        entity_names = self._resolve_entity_names(parsed)
        retrieval_result = self._run_retrieval(
            question, parsed, entity_names, stage_t0, verbose
        )
        if retrieval_result is None:
            empty = self._build_empty_answer(t0, stage_latencies, cache_before)
            empty.detected_language = language
            return empty

        (
            passages_fetched,
            bm25_results,
            vector_results,
            stage_latencies["retrieval"],
        ) = retrieval_result

        self._log_detail(verbose, "BM25: %d passages", len(bm25_results))
        self._log_detail(verbose, "Vector: %d passages", len(vector_results))

        self._log_step(verbose, 3, "Graph enrichment (Wikidata)...")
        stage_t0 = time.perf_counter()
        graph_results = (
            self._get_graph().enrich_all(entity_names)
            if entity_names and self.config.graph_enrichment
            else []
        )
        stage_latencies["graph"] = round(time.perf_counter() - stage_t0, 4)

        self._log_graph_results(verbose, graph_results)

        self._log_step(verbose, 4, "Ranking (RRF + cross-encoder)...")
        stage_t0 = time.perf_counter()
        ranked = self._get_ranker().rank(
            question,
            bm25_results,
            vector_results,
            top_k=self.config.rerank_top_k,
            use_cross_encoder=self.config.cross_encoder_reranking,
        )
        stage_latencies["ranking"] = round(time.perf_counter() - stage_t0, 4)
        passages_reranked = len(ranked)

        self._log_step(verbose, 5, "Extractive answer span extraction...")
        stage_t0 = time.perf_counter()
        all_candidates, extraction_errors = self._collect_candidates(
            parsed,
            ranked,
            verbose=verbose,
        )
        reretrieval_latency = self._per_candidate_reretrieval(
            parsed,
            all_candidates,
            verbose=verbose,
        )
        if reretrieval_latency > 0.0:
            stage_latencies["per_candidate_retrieval"] = reretrieval_latency
        stage_latencies["extraction"] = round(time.perf_counter() - stage_t0, 4)
        passages_extracted = len(all_candidates)

        bidirectional_signal = 0.0
        if self.config.bidirectional_validation and all_candidates:
            stage_t0 = time.perf_counter()
            bidirectional_signal = bidirectional_score(
                all_candidates[0].span,
                question,
                self.dataset_query_engine,
                top_k=3,
            )
            stage_latencies["double_check"] = round(time.perf_counter() - stage_t0, 4)

        configure_entailment_model(self._current_nli_model)

        self._log_step(verbose, 6, "Confidence scoring...")
        stage_t0 = time.perf_counter()
        answer = self._score_answer(
            question,
            parsed,
            ranked,
            all_candidates,
            graph_results,
            bidirectional_signal=bidirectional_signal,
        )
        stage_latencies["scoring"] = round(time.perf_counter() - stage_t0, 4)

        result = self._run_iterative_retrieval(
            question, parsed, answer, graph_results, verbose
        )
        if result is not None:
            answer, latencies, f_ranked, f_errors, f_extracted, f_reranked = result
            stage_latencies.update(latencies)
            final_ranked = f_ranked
            final_extraction_errors = f_errors
            final_passages_extracted = f_extracted
            final_passages_reranked = f_reranked
        else:
            final_ranked = ranked
            final_extraction_errors = extraction_errors
            final_passages_extracted = passages_extracted
            final_passages_reranked = passages_reranked

        total_latency = round(time.perf_counter() - t0, 4)
        stage_latencies["total"] = total_latency
        cache_after = get_cache_metrics_snapshot()
        cache_hits, cache_misses, hits_by_ns, misses_by_ns = self._cache_metrics_delta(
            cache_before, cache_after
        )
        answer.diagnostics = AnswerDiagnostics(
            total_latency_s=total_latency,
            stage_latencies_s=stage_latencies,
            passages_fetched=passages_fetched,
            passages_reranked=final_passages_reranked,
            passages_extracted=final_passages_extracted,
            retrieval_empty=False,
            extraction_errors=final_extraction_errors,
            fallback_answer=self._is_fallback_answer(answer),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hits_by_namespace=hits_by_ns,
            cache_misses_by_namespace=misses_by_ns,
            top_retrieved_passages=[rp.passage.text for rp in final_ranked[:10]],
        )
        answer.detected_language = language

        if verbose:
            self._print_answer(answer, total_latency)

        return answer

    def _generate_queries(self, parsed: ParsedQuestion) -> list[str]:
        return (
            generate_search_queries(
                parsed, augment_context=self.config.query_context_augmentation
            )
            if self.config.query_expansion
            else [parsed.raw]
        )

    def _resolve_entity_names(self, parsed: ParsedQuestion) -> list[str]:
        entity_names = [str(entity["text"]) for entity in parsed.entities]
        if not entity_names:
            _question_words = frozenset(
                {"who", "what", "when", "where", "why", "how", "whom", "whose"}
            )
            nps = [
                nc
                for nc in parsed.noun_chunks
                if nc.lower().strip() not in _question_words
            ]
            if nps:
                entity_names = [max(nps, key=len)]
        return entity_names

    def _build_empty_answer(
        self,
        t0: float,
        stage_latencies: dict[str, float],
        cache_before: CacheMetrics,
    ) -> FinalAnswer:
        total_latency = round(time.perf_counter() - t0, 4)
        cache_after = get_cache_metrics_snapshot()
        cache_hits, cache_misses, hits_by_ns, misses_by_ns = self._cache_metrics_delta(
            cache_before, cache_after
        )
        return FinalAnswer(
            answer="Could not retrieve relevant passages.",
            confidence=0.0,
            source="",
            url="",
            diagnostics=AnswerDiagnostics(
                total_latency_s=total_latency,
                stage_latencies_s=stage_latencies,
                passages_fetched=0,
                passages_reranked=0,
                passages_extracted=0,
                retrieval_empty=True,
                extraction_errors=0,
                fallback_answer=True,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_hits_by_namespace=hits_by_ns,
                cache_misses_by_namespace=misses_by_ns,
                top_retrieved_passages=[],
            ),
        )

    def _run_iterative_retrieval(
        self,
        question: str,
        parsed: ParsedQuestion,
        answer: FinalAnswer,
        graph_results: list[GraphResult],
        verbose: bool,
    ) -> (
        tuple[
            FinalAnswer,
            dict[str, float],
            list[RankedPassage],
            int,
            int,
            int,
        ]
        | None
    ):
        if (
            not self.config.iterative_retrieval
            or answer.confidence >= self.config.iterative_retrieval_threshold
            or answer.answer
            in ("No answer found", "Could not retrieve relevant passages.")
        ):
            return None

        iterative_t0 = time.perf_counter()
        best_answer = answer
        best_confidence = answer.confidence
        improved = False
        final_ranked: list[RankedPassage] = []
        final_errors = 0
        final_extracted = 0
        final_reranked = 0

        for pass_index in range(1, self.config.max_retrieval_passes):
            self._log_detail(verbose, "Iterative retrieval pass %d", pass_index + 1)
            refined_query = f"{question} {best_answer.answer}"
            refined_passages = self._dedupe_passages(
                self.dataset_query_engine.query(
                    refined_query, top_k=self.config.wikipedia_top_k_per_query
                )
            )
            if not refined_passages:
                continue

            result = self._run_single_iterative_pass(
                question,
                parsed,
                refined_query,
                refined_passages,
                graph_results,
                verbose,
            )
            if result is None:
                continue

            refined_answer, refined_ranked, refined_errors, refined_candidates = result
            if refined_answer.confidence > best_confidence:
                best_answer = refined_answer
                best_confidence = refined_answer.confidence
                improved = True
                final_ranked = refined_ranked
                final_errors = refined_errors
                final_extracted = len(refined_candidates)
                final_reranked = len(refined_ranked)

        if not improved:
            return None

        return (
            best_answer,
            {"iterative_retrieval": round(time.perf_counter() - iterative_t0, 4)},
            final_ranked,
            final_errors,
            final_extracted,
            final_reranked,
        )

    def _run_single_iterative_pass(
        self,
        question: str,
        parsed: ParsedQuestion,
        refined_query: str,
        refined_passages: list[Passage],
        graph_results: list[GraphResult],
        verbose: bool,
    ) -> tuple[FinalAnswer, list[RankedPassage], int, list[AnswerCandidate]] | None:
        refined_needs_reindex = self._needs_reindex(refined_passages)
        refined_bm25, refined_vector = self._retrieve_parallel(
            refined_query,
            refined_passages,
            refined_needs_reindex,
            vector_retriever=self._get_vector(),
            vector_enabled=self.config.vector_retrieval,
            top_k=self.config.retrieval_top_k,
        )
        refined_ranked = self._get_ranker().rank(
            question,
            refined_bm25,
            refined_vector,
            top_k=self.config.rerank_top_k,
            use_cross_encoder=self.config.cross_encoder_reranking,
        )
        refined_candidates, refined_errors = self._collect_candidates(
            parsed,
            refined_ranked,
            verbose=verbose,
        )
        self._per_candidate_reretrieval(
            parsed,
            refined_candidates,
            verbose=verbose,
        )
        refined_bidirectional = 0.0
        if self.config.bidirectional_validation and refined_candidates:
            refined_bidirectional = bidirectional_score(
                refined_candidates[0].span,
                question,
                self.dataset_query_engine,
                top_k=3,
            )
        refined_answer = self._score_answer(
            question,
            parsed,
            refined_ranked,
            refined_candidates,
            graph_results,
            bidirectional_signal=refined_bidirectional,
        )
        return refined_answer, refined_ranked, refined_errors, refined_candidates

    def _print_answer(self, answer: FinalAnswer, elapsed: float) -> None:
        logger.info("=" * 50)
        logger.info("  ANSWER:     %s", answer.answer)
        logger.info("  CONFIDENCE: %.1f%%", answer.confidence * 100)
        logger.info("  LANGUAGE:   %s", answer.detected_language or "en")
        logger.info("  SOURCE:     %s", answer.source)
        logger.info("  URL:        %s", answer.url)
        if answer.graph_facts:
            logger.info("  GRAPH CORROBORATION:")
            for f in answer.graph_facts[:3]:
                logger.info("    · %s", f)
        logger.info("  Confidence breakdown:")
        for k, v in answer.confidence_breakdown.items():
            logger.info("    %s: %s", k, v)
        if answer.diagnostics is not None:
            diagnostics = answer.diagnostics
            logger.info("  Diagnostics:")
            logger.info(
                "    passages: fetched=%d reranked=%d extracted=%d",
                diagnostics.passages_fetched,
                diagnostics.passages_reranked,
                diagnostics.passages_extracted,
            )
            logger.info(
                "    cache: hits=%d misses=%d",
                diagnostics.cache_hits,
                diagnostics.cache_misses,
            )
            if diagnostics.stage_latencies_s:
                formatted = ", ".join(
                    f"{stage}={latency:.3f}s"
                    for stage, latency in diagnostics.stage_latencies_s.items()
                )
                logger.info("    timings: %s", formatted)
        logger.info("  Time: %.2fs", elapsed)
        logger.info("=" * 50)
