from transformers import pipeline

from watson_lite.core.models import (
    AnswerCandidate,
    FinalAnswer,
    GraphResult,
    RankedPassage,
)

EXTRACTIVE_MODEL = "deepset/roberta-base-squad2"


class ExtractiveReader:
    def __init__(self, model_name: str = EXTRACTIVE_MODEL) -> None:
        print(f"[Extractor] Loading extractive QA model: {model_name}")
        self.qa = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=-1,
        )

    def extract(
        self, question: str, passages: list[RankedPassage], top_k: int = 5
    ) -> list[AnswerCandidate]:
        candidates = []

        for rp in passages[:top_k]:
            try:
                result = self.qa(
                    question=question,
                    context=rp.passage.text,
                    max_answer_len=100,
                )
                candidates.append(
                    AnswerCandidate(
                        span=result["answer"],
                        source=rp.passage.source,
                        url=rp.passage.url,
                        passage=rp.passage.text,
                        extraction_score=float(result["score"]),
                        rank=rp.rank,
                    )
                )
            except Exception as e:
                print(f"[Extractor] Skipped passage: {e}")

        candidates.sort(key=lambda c: c.extraction_score, reverse=True)
        return candidates


class ConfidenceScorer:
    def score(
        self,
        candidates: list[AnswerCandidate],
        graph_results: list[GraphResult],
        _question_type: str,
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

        extraction_conf = best.extraction_score

        spans = [c.span.lower().strip() for c in candidates]
        agreement = spans.count(best.span.lower().strip()) / len(spans)

        graph_corroborated = False
        graph_facts_used = []
        for gr in graph_results:
            for fact in gr.facts:
                if (
                    best.span.lower() in fact.value.lower()
                    or fact.value.lower() in best.span.lower()
                ):
                    graph_corroborated = True
                    graph_facts_used.append(f"{fact.property_label}: {fact.value}")

        graph_signal = 0.2 if graph_corroborated else 0.0

        rank_signal = max(0.0, 1.0 - (best.rank - 1) * 0.1)

        confidence = (
            0.50 * extraction_conf
            + 0.20 * agreement
            + 0.20 * graph_signal
            + 0.10 * rank_signal
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
                "span_agreement": round(agreement, 3),
                "graph_corroboration": graph_signal,
                "passage_rank_signal": round(rank_signal, 3),
            },
        )
