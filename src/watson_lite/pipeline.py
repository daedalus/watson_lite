import hashlib
import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor

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
    fetch_elasticsearch_passages,
    fetch_wikibooks_passages,
    fetch_wikipedia_page_by_title,
    fetch_wikipedia_passages,
)
from watson_lite.retrieval.dataset_query_engine import (
    DatasetProvider,
    DatasetQueryEngine,
)
from watson_lite.retrieval.query_formulation import generate_search_queries
from watson_lite.retrieval.vector_retriever import VectorRetriever
from watson_lite.scoring.double_check import bidirectional_score

logger = logging.getLogger(__name__)
_NON_WORD = re.compile(r"\W+")


def _passages_hash(passages: list[Passage]) -> str:
    """Return a compact hash that identifies a list of passages by content."""
    parts = "|".join(f"{p.source}:{p.url}:{p.text}" for p in passages)
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
        self.dataset_query_engine = DatasetQueryEngine(
            providers=(
                DatasetProvider("wikipedia", fetch_wikipedia_passages),
                DatasetProvider("wikibooks", fetch_wikibooks_passages),
                DatasetProvider(
                    "elasticsearch",
                    lambda query, *, top_k: fetch_elasticsearch_passages(
                        query,
                        top_k=top_k,
                        base_url=self.config.elasticsearch_url,
                        index=self.config.elasticsearch_index,
                    ),
                ),
            ),
            enabled_datasets=self.config.dataset_sources,
        )
        self.graph: WikidataGraph | None = None
        self.ranker: Ranker | None = None
        self.reader: ExtractiveReader | None = None
        self.scorer = ConfidenceScorer()
        self._last_passage_hash: str | None = None
        logger.info("Core components loaded. Heavy models will load lazily.")

    def _get_nlp(self) -> NLPProcessor:
        if self.nlp is None:
            self.nlp = NLPProcessor(semantic_nlp=self.config.semantic_nlp)
        return self.nlp

    def _get_vector(self) -> VectorRetriever | None:
        if not self.config.vector_retrieval:
            return None
        if self.vector is None:
            self.vector = VectorRetriever()
        return self.vector

    def _get_graph(self) -> WikidataGraph:
        if self.graph is None:
            self.graph = WikidataGraph()
        return self.graph

    def _get_ranker(self) -> Ranker:
        if self.ranker is None:
            self.ranker = Ranker(
                enable_cross_encoder=self.config.cross_encoder_reranking
            )
        return self.ranker

    def _get_reader(self) -> ExtractiveReader:
        if self.reader is None:
            self.reader = ExtractiveReader(device=self.device)
        return self.reader

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
            return vector_retriever.retrieve(question, top_k=top_k)

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
        return is_fallback_answer_text(answer.answer)

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
                        extraction_score=0.3,
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
            parsed.question_type,
            lat_qids=parsed.lat_qids,
            question=question,
            ranked_passages=ranked,
            enable_question_type_bonus=self.config.question_type_bonus,
            enable_type_coercion=self.config.type_coercion,
            enable_term_match=self.config.term_match,
            enable_consistency=self.config.consistency,
            enable_answer_merging=self.config.answer_merging,
            bidirectional_signal=bidirectional_signal,
        )

    def answer(  # pylint: disable=too-many-statements,too-many-locals
        self, question: str, verbose: bool = True
    ) -> FinalAnswer:
        if not question:
            raise ValueError("question must not be empty")

        t0 = time.perf_counter()
        stage_latencies: dict[str, float] = {}
        cache_before = get_cache_metrics_snapshot()

        self._log_step(verbose, 1, "NLP preprocessing...")
        stage_t0 = time.perf_counter()
        parsed = self._get_nlp().process(
            question, semantic_nlp=self.config.semantic_nlp
        )
        stage_latencies["nlp"] = round(time.perf_counter() - stage_t0, 4)
        self._log_detail(verbose, "Type: %s", parsed.question_type)
        self._log_detail(verbose, "Entities: %s", [e["text"] for e in parsed.entities])
        self._log_detail(verbose, "Sub-questions: %s", parsed.sub_questions)

        self._log_step(verbose, 2, "Parallel retrieval (BM25 + Vector)...")
        stage_t0 = time.perf_counter()
        queries = (
            generate_search_queries(
                parsed, augment_context=self.config.query_context_augmentation
            )
            if self.config.query_expansion
            else [parsed.raw]
        )
        self._log_detail(verbose, "Search queries: %s", queries)

        all_passages: list[Passage] = []
        for query in queries:
            all_passages.extend(
                self.dataset_query_engine.query(
                    query, top_k=self.config.wikipedia_top_k_per_query
                )
            )

        entity_names = [str(entity["text"]) for entity in parsed.entities]
        if entity_names:
            self._log_detail(verbose, "Entity direct page fetch: %s", entity_names)
            for entity_text in entity_names:
                all_passages.extend(fetch_wikipedia_page_by_title(entity_text))

        passages = self._dedupe_passages(all_passages)

        if not passages:
            total_latency = round(time.perf_counter() - t0, 4)
            cache_after = get_cache_metrics_snapshot()
            cache_hits, cache_misses, hits_by_ns, misses_by_ns = (
                self._cache_metrics_delta(cache_before, cache_after)
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

        passages_fetched = len(passages)
        passage_hash = _passages_hash(passages)
        needs_reindex = passage_hash != self._last_passage_hash
        self._last_passage_hash = passage_hash

        bm25_results, vector_results = self._retrieve_parallel(
            question,
            passages,
            needs_reindex,
            vector_retriever=self._get_vector(),
            vector_enabled=self.config.vector_retrieval,
            top_k=self.config.retrieval_top_k,
        )
        stage_latencies["retrieval"] = round(time.perf_counter() - stage_t0, 4)

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

        final_ranked = ranked
        final_extraction_errors = extraction_errors
        final_passages_extracted = passages_extracted
        final_passages_reranked = passages_reranked

        if (
            self.config.iterative_retrieval
            and answer.confidence < self.config.iterative_retrieval_threshold
            and answer.answer
            not in ("No answer found", "Could not retrieve relevant passages.")
        ):
            iterative_t0 = time.perf_counter()
            best_answer = answer
            best_confidence = answer.confidence
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

                refined_hash = _passages_hash(refined_passages)
                refined_needs_reindex = refined_hash != self._last_passage_hash
                self._last_passage_hash = refined_hash
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
                if refined_answer.confidence > best_confidence:
                    best_answer = refined_answer
                    best_confidence = refined_answer.confidence
                    final_ranked = refined_ranked
                    final_extraction_errors = refined_errors
                    final_passages_extracted = len(refined_candidates)
                    final_passages_reranked = len(refined_ranked)
            answer = best_answer
            stage_latencies["iterative_retrieval"] = round(
                time.perf_counter() - iterative_t0, 4
            )

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

        if verbose:
            self._print_answer(answer, total_latency)

        return answer

    def _print_answer(self, answer: FinalAnswer, elapsed: float) -> None:
        logger.info("=" * 50)
        logger.info("  ANSWER:     %s", answer.answer)
        logger.info("  CONFIDENCE: %.1f%%", answer.confidence * 100)
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
