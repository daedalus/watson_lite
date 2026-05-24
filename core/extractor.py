"""
core/extractor.py
Extractive QA — finds answer spans in passages.
Uses pretrained roberta-base-squad2 (no generation, no LLM).
Also computes confidence score from evidence signals.
"""

from transformers import pipeline
from dataclasses import dataclass, field
from typing import List, Optional
from ranking.ranker import RankedPassage
from graph.wikidata import GraphResult


EXTRACTIVE_MODEL = "deepset/roberta-base-squad2"


@dataclass
class AnswerCandidate:
    span: str                       # extracted text span
    source: str                     # article title
    url: str
    passage: str                    # full passage for context
    extraction_score: float         # model's span confidence
    rank: int                       # passage rank
    graph_corroborated: bool = False


@dataclass
class FinalAnswer:
    answer: str
    confidence: float               # 0.0 – 1.0
    source: str
    url: str
    supporting_passages: List[str] = field(default_factory=list)
    graph_facts: List[str] = field(default_factory=list)
    confidence_breakdown: dict = field(default_factory=dict)


class ExtractiveReader:
    def __init__(self, model_name: str = EXTRACTIVE_MODEL):
        print(f"[Extractor] Loading extractive QA model: {model_name}")
        self.qa = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=-1,              # CPU
        )

    def extract(self, question: str, passages: List[RankedPassage], top_k: int = 5) -> List[AnswerCandidate]:
        candidates = []

        for rp in passages[:top_k]:
            try:
                result = self.qa(
                    question=question,
                    context=rp.passage.text,
                    max_answer_len=100,
                )
                candidates.append(AnswerCandidate(
                    span=result["answer"],
                    source=rp.passage.source,
                    url=rp.passage.url,
                    passage=rp.passage.text,
                    extraction_score=float(result["score"]),
                    rank=rp.rank,
                ))
            except Exception as e:
                print(f"[Extractor] Skipped passage: {e}")

        # Sort by extraction confidence
        candidates.sort(key=lambda c: c.extraction_score, reverse=True)
        return candidates


class ConfidenceScorer:
    """
    Watson-inspired confidence scoring.
    Combines multiple signals without any trained weights.
    """

    def score(
        self,
        candidates: List[AnswerCandidate],
        graph_results: List[GraphResult],
        question_type: str,
    ) -> FinalAnswer:

        if not candidates:
            return FinalAnswer(
                answer="No answer found",
                confidence=0.0,
                source="",
                url="",
                confidence_breakdown={"reason": "no candidates"},
            )

        best = candidates[0]

        # Signal 1: Extraction model confidence (0–1)
        extraction_conf = best.extraction_score

        # Signal 2: Answer span agreement across passages
        spans = [c.span.lower().strip() for c in candidates]
        agreement = spans.count(best.span.lower().strip()) / len(spans)

        # Signal 3: Graph corroboration — does graph mention this span?
        graph_corroborated = False
        graph_facts_used = []
        for gr in graph_results:
            for fact in gr.facts:
                if best.span.lower() in fact.value.lower() or fact.value.lower() in best.span.lower():
                    graph_corroborated = True
                    graph_facts_used.append(f"{fact.property_label}: {fact.value}")

        graph_signal = 0.2 if graph_corroborated else 0.0

        # Signal 4: Passage rank bonus (top passage = higher conf)
        rank_signal = max(0.0, 1.0 - (best.rank - 1) * 0.1)

        # Weighted combination (no training — weights are heuristic)
        confidence = (
            0.50 * extraction_conf +
            0.20 * agreement +
            0.20 * graph_signal +
            0.10 * rank_signal
        )
        confidence = round(min(confidence, 1.0), 3)

        return FinalAnswer(
            answer=best.span,
            confidence=confidence,
            source=best.source,
            url=best.url,
            supporting_passages=[c.passage[:200] for c in candidates[:3]],
            graph_facts=graph_facts_used[:5],
            confidence_breakdown={
                "extraction_model": round(extraction_conf, 3),
                "span_agreement":   round(agreement, 3),
                "graph_corroboration": graph_signal,
                "passage_rank_signal": round(rank_signal, 3),
            },
        )


if __name__ == "__main__":
    # Quick smoke test
    from ranking.ranker import RankedPassage
    from retrieval.bm25_retriever import Passage

    p = Passage(
        text="The Eiffel Tower was designed by Gustave Eiffel and built between 1887 and 1889.",
        source="Eiffel Tower",
        url="https://en.wikipedia.org/wiki/Eiffel_Tower",
        score=0.9,
        rank=1,
    )
    rp = RankedPassage(passage=p, final_score=0.9, rank=1)

    reader = ExtractiveReader()
    candidates = reader.extract("Who designed the Eiffel Tower?", [rp])
    print(f"Candidate span: {candidates[0].span}")
    print(f"Confidence:     {candidates[0].extraction_score:.3f}")
