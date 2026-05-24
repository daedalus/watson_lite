from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Literal, overload

from transformers import pipeline

from watson_lite.core.models import (
    AnswerCandidate,
    FinalAnswer,
    GraphResult,
    RankedPassage,
)
from watson_lite.scoring.answer_merging import merge_candidates_by_qid
from watson_lite.scoring.consistency import (
    score_geospatial_consistency,
    score_temporal_consistency,
)
from watson_lite.scoring.term_match import score_term_match
from watson_lite.scoring.type_coercion import score_type_coercion

if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)

EXTRACTIVE_MODEL = "deepset/roberta-base-squad2"

# Patterns used to apply a small question-type-aware confidence bonus.
_MULTI_WORD_CAP = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_YEAR = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")
_DATE_WORD = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|\d{1,2}\s+\w+|\w+\s+\d{4})\b"
)


def _question_type_bonus(span: str, question_type: str) -> float:
    """Return a small bonus in [0, 0.1] when the span form matches the question type."""
    if question_type == "who" and _MULTI_WORD_CAP.search(span):
        return 0.1
    if question_type == "when" and (_YEAR.search(span) or _DATE_WORD.search(span)):
        return 0.1
    return 0.0


class ExtractiveReader:
    def __init__(self, model_name: str = EXTRACTIVE_MODEL) -> None:
        logger.debug("Loading extractive QA model: %s", model_name)
        self.qa: Pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=-1,
        )

    @overload
    def extract(
        self,
        question: str,
        passages: list[RankedPassage],
        top_k: int = 5,
        return_stats: Literal[False] = False,
    ) -> list[AnswerCandidate]:
        pass

    @overload
    def extract(
        self,
        question: str,
        passages: list[RankedPassage],
        top_k: int = 5,
        return_stats: Literal[True] = True,
    ) -> tuple[list[AnswerCandidate], int]:
        pass

    def extract(
        self,
        question: str,
        passages: list[RankedPassage],
        top_k: int = 5,
        return_stats: bool = False,
    ) -> list[AnswerCandidate] | tuple[list[AnswerCandidate], int]:
        candidates = []
        extraction_errors = 0

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
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning("Skipped passage due to extraction error: %s", e)
                extraction_errors += 1

        candidates.sort(key=lambda c: c.extraction_score, reverse=True)
        if return_stats:
            return candidates, extraction_errors
        return candidates


class ConfidenceScorer:
    def score(  # pylint: disable=too-many-arguments
        self,
        candidates: list[AnswerCandidate],
        graph_results: list[GraphResult],
        question_type: str,
        lat_qids: list[str] | None = None,
        *,
        question: str = "",
        ranked_passages: list[RankedPassage] | None = None,
        enable_question_type_bonus: bool = True,
        enable_type_coercion: bool = True,
        enable_term_match: bool = True,
        enable_consistency: bool = True,
        enable_answer_merging: bool = True,
    ) -> FinalAnswer:

        if not candidates:
            return FinalAnswer(
                answer="No answer found",
                confidence=0.0,
                source="",
                url="",
                confidence_breakdown={"reason": "no candidates"},
            )

        if enable_answer_merging:
            candidates = merge_candidates_by_qid(candidates)

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

        qt_bonus = (
            _question_type_bonus(best.span, question_type)
            if enable_question_type_bonus
            else 0.0
        )

        type_signal = (
            score_type_coercion(candidates, lat_qids or [])
            if enable_type_coercion
            else 0.0
        )

        term_match_signal = (
            score_term_match(question, ranked_passages)
            if enable_term_match and ranked_passages
            else 0.0
        )

        temporal_signal = (
            score_temporal_consistency(candidates, graph_results)
            if enable_consistency
            else 0.0
        )
        geo_signal = (
            score_geospatial_consistency(candidates, graph_results)
            if enable_consistency
            else 0.0
        )

        confidence = (
            0.35 * extraction_conf
            + 0.10 * agreement
            + 0.15 * graph_signal
            + 0.10 * rank_signal
            + qt_bonus
            + 0.15 * type_signal
            + 0.10 * term_match_signal
            + 0.05 * temporal_signal
            + 0.05 * geo_signal
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
                "question_type_bonus": qt_bonus,
                "type_coercion": type_signal,
                "term_match": round(term_match_signal, 3),
                "temporal_consistency": temporal_signal,
                "geospatial_consistency": geo_signal,
            },
        )
