import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from watson_lite.core.extractor import ConfidenceScorer, ExtractiveReader
from watson_lite.core.models import FinalAnswer, GraphResult, Passage
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

    def answer(self, question: str, verbose: bool = True) -> FinalAnswer:
        if not question:
            raise ValueError("question must not be empty")

        t0 = time.time()

        self._log_step(verbose, 1, "NLP preprocessing...")
        parsed = self.nlp.process(question)
        self._log_detail(verbose, "Type: %s", parsed.question_type)
        self._log_detail(verbose, "Entities: %s", [e["text"] for e in parsed.entities])
        self._log_detail(verbose, "Sub-questions: %s", parsed.sub_questions)

        self._log_step(verbose, 2, "Parallel retrieval (BM25 + Vector)...")
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
            return FinalAnswer(
                answer="Could not retrieve relevant passages.",
                confidence=0.0,
                source="",
                url="",
            )

        passage_hash = _passages_hash(passages)
        needs_reindex = passage_hash != self._last_passage_hash
        self._last_passage_hash = passage_hash

        bm25_results, vector_results = self._retrieve_parallel(
            question, passages, needs_reindex
        )

        self._log_detail(verbose, "BM25: %d passages", len(bm25_results))
        self._log_detail(verbose, "Vector: %d passages", len(vector_results))

        self._log_step(verbose, 3, "Graph enrichment (Wikidata)...")
        entity_names = [str(e["text"]) for e in parsed.entities]
        graph_results = self.graph.enrich_all(entity_names) if entity_names else []

        self._log_graph_results(verbose, graph_results)

        self._log_step(verbose, 4, "Ranking (RRF + cross-encoder)...")
        ranked = self.ranker.rank(question, bm25_results, vector_results, top_k=10)

        self._log_step(verbose, 5, "Extractive answer span extraction...")

        all_candidates = []
        for sub_q in parsed.sub_questions:
            candidates = self.reader.extract(sub_q, ranked, top_k=5)
            all_candidates.extend(candidates)

        all_candidates.sort(key=lambda c: c.extraction_score, reverse=True)

        self._log_step(verbose, 6, "Confidence scoring...")
        answer = self.scorer.score(
            all_candidates,
            graph_results,
            parsed.question_type,
            lat_qids=parsed.lat_qids,
        )

        elapsed = time.time() - t0
        if verbose:
            self._print_answer(answer, elapsed)

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
