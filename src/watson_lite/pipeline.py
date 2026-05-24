import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from watson_lite.core.cache import CacheMetrics, get_cache_metrics_snapshot
from watson_lite.core.extractor import ConfidenceScorer, ExtractiveReader
from watson_lite.core.models import AnswerDiagnostics, FinalAnswer, GraphResult, Passage
from watson_lite.core.nlp import NLPProcessor
from watson_lite.graph.wikidata import WikidataGraph
from watson_lite.ranking.ranker import Ranker
from watson_lite.retrieval.bm25_retriever import BM25Retriever, fetch_wikipedia_passages
from watson_lite.retrieval.query_formulation import generate_search_queries
from watson_lite.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def _passages_hash(passages: list[Passage]) -> str:
    """Return a compact hash that identifies a list of passages by content."""
    parts = "|".join(f"{p.source}:{p.url}:{p.text}" for p in passages)
    return hashlib.sha256(parts.encode()).hexdigest()


class WatsonLite:
    def __init__(self) -> None:
        logger.info("WatsonLite — Initializing pipeline")
        self.nlp = NLPProcessor()
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever()
        self.graph = WikidataGraph()
        self.ranker = Ranker()
        self.reader = ExtractiveReader()
        self.scorer = ConfidenceScorer()
        self._last_passage_hash: str | None = None
        logger.info("All components loaded. Ready.")

    def _retrieve_parallel(
        self,
        question: str,
        passages: list[Passage],
        needs_reindex: bool,
    ) -> tuple[list[Passage], list[Passage]]:
        """Run BM25 and vector retrieval in parallel, re-indexing only when needed."""

        def _bm25_work() -> list[Passage]:
            if needs_reindex:
                self.bm25.index(passages)
            return self.bm25.retrieve(question, top_k=20)

        def _vector_work() -> list[Passage]:
            if needs_reindex:
                self.vector.index_passages(passages)
            return self.vector.retrieve(question, top_k=20)

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
        return answer.answer in {
            "No answer found",
            "Could not retrieve relevant passages.",
        }

    def answer(  # pylint: disable=too-many-statements
        self, question: str, verbose: bool = True
    ) -> FinalAnswer:
        if not question:
            raise ValueError("question must not be empty")

        t0 = time.perf_counter()
        stage_latencies: dict[str, float] = {}
        cache_before = get_cache_metrics_snapshot()

        self._log_step(verbose, 1, "NLP preprocessing...")
        stage_t0 = time.perf_counter()
        parsed = self.nlp.process(question)
        stage_latencies["nlp"] = round(time.perf_counter() - stage_t0, 4)
        self._log_detail(verbose, "Type: %s", parsed.question_type)
        self._log_detail(verbose, "Entities: %s", [e["text"] for e in parsed.entities])
        self._log_detail(verbose, "Sub-questions: %s", parsed.sub_questions)

        self._log_step(verbose, 2, "Parallel retrieval (BM25 + Vector)...")
        stage_t0 = time.perf_counter()
        queries = generate_search_queries(parsed)
        self._log_detail(verbose, "Search queries: %s", queries)

        seen_texts: set[str] = set()
        all_passages: list[Passage] = []
        for q in queries:
            for p in fetch_wikipedia_passages(q, top_k=5):
                if p.text not in seen_texts:
                    seen_texts.add(p.text)
                    all_passages.append(p)

        passages = all_passages

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
            question, passages, needs_reindex
        )
        stage_latencies["retrieval"] = round(time.perf_counter() - stage_t0, 4)

        self._log_detail(verbose, "BM25: %d passages", len(bm25_results))
        self._log_detail(verbose, "Vector: %d passages", len(vector_results))

        self._log_step(verbose, 3, "Graph enrichment (Wikidata)...")
        stage_t0 = time.perf_counter()
        entity_names = [str(e["text"]) for e in parsed.entities]
        graph_results = self.graph.enrich_all(entity_names) if entity_names else []
        stage_latencies["graph"] = round(time.perf_counter() - stage_t0, 4)

        self._log_graph_results(verbose, graph_results)

        self._log_step(verbose, 4, "Ranking (RRF + cross-encoder)...")
        stage_t0 = time.perf_counter()
        ranked = self.ranker.rank(question, bm25_results, vector_results, top_k=10)
        stage_latencies["ranking"] = round(time.perf_counter() - stage_t0, 4)
        passages_reranked = len(ranked)

        self._log_step(verbose, 5, "Extractive answer span extraction...")
        stage_t0 = time.perf_counter()

        all_candidates = []
        extraction_errors = 0
        for sub_q in parsed.sub_questions:
            extraction_result = self.reader.extract(
                sub_q, ranked, top_k=5, return_stats=True
            )
            if isinstance(extraction_result, tuple):
                candidates, errors = extraction_result
            else:
                candidates, errors = extraction_result, 0
            all_candidates.extend(candidates)
            extraction_errors += errors

        all_candidates.sort(key=lambda c: c.extraction_score, reverse=True)
        stage_latencies["extraction"] = round(time.perf_counter() - stage_t0, 4)
        passages_extracted = len(all_candidates)

        self._log_step(verbose, 6, "Confidence scoring...")
        stage_t0 = time.perf_counter()
        answer = self.scorer.score(
            all_candidates,
            graph_results,
            parsed.question_type,
            lat_qids=parsed.lat_qids,
        )
        stage_latencies["scoring"] = round(time.perf_counter() - stage_t0, 4)

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
            passages_reranked=passages_reranked,
            passages_extracted=passages_extracted,
            retrieval_empty=False,
            extraction_errors=extraction_errors,
            fallback_answer=self._is_fallback_answer(answer),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hits_by_namespace=hits_by_ns,
            cache_misses_by_namespace=misses_by_ns,
            top_retrieved_passages=[rp.passage.text for rp in ranked[:10]],
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
        logger.info("  Time: %.2fs", elapsed)
        logger.info("=" * 50)
