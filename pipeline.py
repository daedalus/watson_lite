"""
pipeline.py
WatsonLite — Main orchestrator.
Connects all stages end-to-end. No LLM. No trained weights of our own.
"""

from core.nlp import NLPProcessor
from retrieval.bm25_retriever import BM25Retriever, fetch_wikipedia_passages
from retrieval.vector_retriever import VectorRetriever
from graph.wikidata import WikidataGraph
from ranking.ranker import Ranker
from core.extractor import ExtractiveReader, ConfidenceScorer, FinalAnswer

import time


class WatsonLite:
    def __init__(self):
        print("=" * 50)
        print("  WatsonLite — Initializing pipeline")
        print("=" * 50)
        self.nlp        = NLPProcessor()
        self.bm25       = BM25Retriever()
        self.vector     = VectorRetriever()
        self.graph      = WikidataGraph()
        self.ranker     = Ranker()
        self.reader     = ExtractiveReader()
        self.scorer     = ConfidenceScorer()
        print("=" * 50)
        print("  All components loaded. Ready.")
        print("=" * 50)

    def answer(self, question: str, verbose: bool = True) -> FinalAnswer:
        t0 = time.time()

        # ── Stage 1: NLP Preprocessing ──────────────────────────────
        if verbose: print(f"\n[1/6] NLP preprocessing...")
        parsed = self.nlp.process(question)
        if verbose:
            print(f"      Type: {parsed.question_type}")
            print(f"      Entities: {[e['text'] for e in parsed.entities]}")
            print(f"      Sub-questions: {parsed.sub_questions}")

        # ── Stage 2: Parallel Retrieval ─────────────────────────────
        if verbose: print(f"\n[2/6] Parallel retrieval (BM25 + Vector)...")
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
            print(f"      BM25: {len(bm25_results)} passages")
            print(f"      Vector: {len(vector_results)} passages")

        # ── Stage 3: Graph Enrichment ────────────────────────────────
        if verbose: print(f"\n[3/6] Graph enrichment (Wikidata)...")
        entity_names = [e["text"] for e in parsed.entities]
        graph_results = self.graph.enrich_all(entity_names) if entity_names else []

        if verbose:
            for gr in graph_results:
                print(f"      {gr.entity_name}: {len(gr.facts)} facts")

        # ── Stage 4: RRF + Cross-encoder Ranking ────────────────────
        if verbose: print(f"\n[4/6] Ranking (RRF + cross-encoder)...")
        ranked = self.ranker.rank(question, bm25_results, vector_results, top_k=10)

        # ── Stage 5: Multi-hypothesis extraction ────────────────────
        if verbose: print(f"\n[5/6] Extractive answer span extraction...")

        # Run extraction on each sub-question (multi-hypothesis)
        all_candidates = []
        for sub_q in parsed.sub_questions:
            candidates = self.reader.extract(sub_q, ranked, top_k=5)
            all_candidates.extend(candidates)

        all_candidates.sort(key=lambda c: c.extraction_score, reverse=True)

        # ── Stage 6: Confidence scoring ─────────────────────────────
        if verbose: print(f"\n[6/6] Confidence scoring...")
        answer = self.scorer.score(all_candidates, graph_results, parsed.question_type)

        elapsed = time.time() - t0
        if verbose:
            self._print_answer(answer, elapsed)

        return answer

    def _print_answer(self, answer: FinalAnswer, elapsed: float):
        print("\n" + "=" * 50)
        print(f"  ANSWER:     {answer.answer}")
        print(f"  CONFIDENCE: {answer.confidence:.1%}")
        print(f"  SOURCE:     {answer.source}")
        print(f"  URL:        {answer.url}")
        if answer.graph_facts:
            print(f"  GRAPH CORROBORATION:")
            for f in answer.graph_facts[:3]:
                print(f"    · {f}")
        print(f"\n  Confidence breakdown:")
        for k, v in answer.confidence_breakdown.items():
            print(f"    {k}: {v}")
        print(f"\n  Time: {elapsed:.2f}s")
        print("=" * 50)


def main():
    watson = WatsonLite()

    questions = [
        "Who designed the Eiffel Tower?",
        "When was the Eiffel Tower built?",
        "What is the height of the Eiffel Tower?",
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"  QUESTION: {q}")
        watson.answer(q)
        print()


if __name__ == "__main__":
    main()
