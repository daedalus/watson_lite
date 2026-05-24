import logging
import time

from watson_lite.core.extractor import ConfidenceScorer, ExtractiveReader
from watson_lite.core.models import FinalAnswer
from watson_lite.core.nlp import NLPProcessor
from watson_lite.graph.wikidata import WikidataGraph
from watson_lite.ranking.ranker import Ranker
from watson_lite.retrieval.bm25_retriever import BM25Retriever, fetch_wikipedia_passages
from watson_lite.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


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
        logger.info("All components loaded. Ready.")

    def answer(self, question: str, verbose: bool = True) -> FinalAnswer:
        if not question:
            raise ValueError("question must not be empty")

        t0 = time.time()

        if verbose:
            logger.info("[1/6] NLP preprocessing...")
        parsed = self.nlp.process(question)
        if verbose:
            logger.info("      Type: %s", parsed.question_type)
            logger.info("      Entities: %s", [e["text"] for e in parsed.entities])
            logger.info("      Sub-questions: %s", parsed.sub_questions)

        if verbose:
            logger.info("[2/6] Parallel retrieval (BM25 + Vector)...")
        passages = fetch_wikipedia_passages(question, top_k=5)

        if not passages:
            return FinalAnswer(
                answer="Could not retrieve relevant passages.",
                confidence=0.0,
                source="",
                url="",
            )

        self.bm25.index(passages)
        bm25_results = self.bm25.retrieve(question, top_k=20)

        self.vector.index_passages(passages)
        vector_results = self.vector.retrieve(question, top_k=20)

        if verbose:
            logger.info("      BM25: %d passages", len(bm25_results))
            logger.info("      Vector: %d passages", len(vector_results))

        if verbose:
            logger.info("[3/6] Graph enrichment (Wikidata)...")
        entity_names = [str(e["text"]) for e in parsed.entities]
        graph_results = self.graph.enrich_all(entity_names) if entity_names else []

        if verbose:
            for gr in graph_results:
                logger.info("      %s: %d facts", gr.entity_name, len(gr.facts))

        if verbose:
            logger.info("[4/6] Ranking (RRF + cross-encoder)...")
        ranked = self.ranker.rank(question, bm25_results, vector_results, top_k=10)

        if verbose:
            logger.info("[5/6] Extractive answer span extraction...")

        all_candidates = []
        for sub_q in parsed.sub_questions:
            candidates = self.reader.extract(sub_q, ranked, top_k=5)
            all_candidates.extend(candidates)

        all_candidates.sort(key=lambda c: c.extraction_score, reverse=True)

        if verbose:
            logger.info("[6/6] Confidence scoring...")
        answer = self.scorer.score(all_candidates, graph_results, parsed.question_type)

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
