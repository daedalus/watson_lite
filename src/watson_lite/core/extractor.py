from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal, overload

try:
    from transformers import pipeline as _hf_pipeline
except ImportError as exc:  # pragma: no cover - exercised via lazy init tests
    hf_pipeline: Any | None = None
    _TRANSFORMERS_IMPORT_ERROR: ImportError | None = exc
else:
    hf_pipeline = _hf_pipeline
    _TRANSFORMERS_IMPORT_ERROR = None

from watson_lite.core.models import (
    AnswerCandidate,
    EvidenceItem,
    FinalAnswer,
    GraphResult,
    RankedPassage,
)
from watson_lite.scoring.answer_merging import merge_candidates_by_qid
from watson_lite.scoring.consistency import (
    score_geospatial_consistency,
    score_temporal_consistency,
)
from watson_lite.scoring.entailment import score_entailment
from watson_lite.scoring.term_match import score_term_match
from watson_lite.scoring.type_coercion import resolve_span_to_qid, score_type_coercion

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
_ENTAILMENT_TOP_CANDIDATES = 3


def _question_word_bonus(span: str, question_word_type: str | None) -> float:
    """Return a small bonus in [0, 0.1] when the span matches the question word type.

    ``question_word_type`` is derived from spaCy POS/morph features
    (``"person"`` for interrogative PRONs, ``"time"`` for interrogative
    ADV/SCONJ).
    """
    if question_word_type == "person" and _MULTI_WORD_CAP.search(span):
        return 0.1
    if question_word_type == "time" and (_YEAR.search(span) or _DATE_WORD.search(span)):
        return 0.1
    return 0.0


class ExtractiveReader:
    def __init__(self, model_name: str = EXTRACTIVE_MODEL, device: int = -1) -> None:
        if hf_pipeline is None:
            raise ImportError(
                "Extractive reading requires transformers (and torch). "
                "Install watson-lite with the 'reader' or 'full' extra."
            ) from _TRANSFORMERS_IMPORT_ERROR
        logger.debug("Loading extractive QA model: %s (device=%d)", model_name, device)
        self.qa: Pipeline = hf_pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device,
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
    def __init__(self, confidence_threshold: float | None = None) -> None:
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _build_evidence_chain(
        candidates: list[AnswerCandidate],
        graph_facts_used: list[str],
        best_span: str,
    ) -> list[EvidenceItem]:
        evidence_chain: list[EvidenceItem] = []

        for candidate in candidates[:3]:
            sentences = candidate.passage.split(". ") if candidate.passage else [""]
            sentence = next(
                (part for part in sentences if candidate.span.lower() in part.lower()),
                sentences[0],
            )
            span_start = sentence.find(candidate.span)
            if span_start < 0:
                span_start = sentence.lower().find(candidate.span.lower())
            span_start = max(0, span_start)
            span_end = span_start + len(candidate.span)
            evidence_chain.append(
                EvidenceItem(
                    passage_text=candidate.passage,
                    sentence=sentence,
                    span=candidate.span,
                    span_start=span_start,
                    span_end=span_end,
                )
            )

        for fact_string in graph_facts_used:
            graph_property = (
                fact_string.split(":", 1)[0] if ":" in fact_string else None
            )
            evidence_chain.append(
                EvidenceItem(
                    passage_text=fact_string,
                    sentence=fact_string,
                    span=best_span,
                    span_start=0,
                    span_end=len(best_span),
                    graph_property=graph_property,
                )
            )

        return evidence_chain

    def _compute_graph_signals(
        self, best: AnswerCandidate, graph_results: list[GraphResult]
    ) -> tuple[bool, list[str]]:
        graph_corroborated = False
        graph_facts_used: list[str] = []
        for gr in graph_results:
            for fact in gr.facts:
                if (
                    best.span.lower() in fact.value.lower()
                    or fact.value.lower() in best.span.lower()
                ):
                    graph_corroborated = True
                    graph_facts_used.append(f"{fact.property_label}: {fact.value}")
        return graph_corroborated, graph_facts_used

    @staticmethod
    def _compute_base_signals(
        best: AnswerCandidate, candidates: list[AnswerCandidate]
    ) -> tuple[float, float, float, float]:
        extraction_conf = best.extraction_score
        spans = [c.span.lower().strip() for c in candidates]
        agreement = spans.count(best.span.lower().strip()) / len(spans)
        rank_signal = max(0.0, 1.0 - (best.rank - 1) * 0.1)
        max_doc_frequency = max(c.doc_frequency for c in candidates)
        frequency_signal = best.doc_frequency / max(1, max_doc_frequency)
        return extraction_conf, agreement, rank_signal, frequency_signal

    @staticmethod
    def _compute_type_and_bonus(
        best: AnswerCandidate,
        candidates: list[AnswerCandidate],
        lat_qids: list[str],
        enable_type_coercion: bool,
        enable_answer_merging: bool,
        enable_question_type_bonus: bool,
        question_word_type: str | None,
    ) -> tuple[str | None, float, float]:
        best_qid = (
            resolve_span_to_qid(best.span)
            if enable_type_coercion or enable_answer_merging
            else None
        )
        type_signal = (
            score_type_coercion(candidates, lat_qids, candidate_qid=best_qid)
            if enable_type_coercion
            else 0.0
        )
        qt_bonus = (
            _question_word_bonus(best.span, question_word_type)
            if enable_question_type_bonus
            else 0.0
        )
        if (
            enable_question_type_bonus
            and enable_type_coercion
            and lat_qids
            and type_signal == 0.0
        ):
            qt_bonus = 0.0
        return best_qid, type_signal, qt_bonus

    @staticmethod
    def _compute_consistency_signals(
        candidates: list[AnswerCandidate],
        graph_results: list[GraphResult],
        enable_consistency: bool,
    ) -> tuple[float, float]:
        temporal = (
            score_temporal_consistency(candidates, graph_results)
            if enable_consistency
            else 0.0
        )
        geo = (
            score_geospatial_consistency(candidates, graph_results)
            if enable_consistency
            else 0.0
        )
        return temporal, geo

    @staticmethod
    def _compute_term_match_signal(
        question: str,
        ranked_passages: list[RankedPassage] | None,
        enable_term_match: bool,
    ) -> float:
        return (
            score_term_match(question, ranked_passages)
            if enable_term_match and ranked_passages
            else 0.0
        )

    @staticmethod
    def _compute_entailment_signal(
        question: str,
        best: AnswerCandidate,
        candidates: list[AnswerCandidate],
        enable_entailment: bool,
    ) -> float:
        if not enable_entailment:
            return 0.0
        top_passages = [
            candidate.passage.strip()
            for candidate in candidates[:_ENTAILMENT_TOP_CANDIDATES]
            if candidate.passage.strip()
        ]
        if not top_passages:
            return 0.0
        return score_entailment(
            question,
            best.span,
            top_passages,
        )

    def _compute_all_signals(
        self,
        best: AnswerCandidate,
        candidates: list[AnswerCandidate],
        graph_results: list[GraphResult],
        question_word_type: str | None,
        lat_qids: list[str],
        question: str,
        ranked_passages: list[RankedPassage] | None,
        enable_type_coercion: bool,
        enable_answer_merging: bool,
        enable_question_type_bonus: bool,
        enable_term_match: bool,
        enable_consistency: bool,
        enable_entailment: bool,
        bidirectional_signal: float,
    ) -> dict[str, Any]:
        extraction_conf, agreement, rank_signal, frequency_signal = (
            self._compute_base_signals(best, candidates)
        )
        graph_corroborated, graph_facts_used = self._compute_graph_signals(
            best, graph_results
        )
        best_qid, type_signal, qt_bonus = self._compute_type_and_bonus(
            best,
            candidates,
            lat_qids,
            enable_type_coercion,
            enable_answer_merging,
            enable_question_type_bonus,
            question_word_type,
        )
        term_match_signal = self._compute_term_match_signal(
            question, ranked_passages, enable_term_match
        )
        entailment_signal = self._compute_entailment_signal(
            question, best, candidates, enable_entailment
        )
        temporal_signal, geo_signal = self._compute_consistency_signals(
            candidates, graph_results, enable_consistency
        )
        return {
            "extraction_conf": extraction_conf,
            "agreement": agreement,
            "rank_signal": rank_signal,
            "frequency_signal": frequency_signal,
            "graph_signal": 0.2 if graph_corroborated else 0.0,
            "graph_facts_used": graph_facts_used,
            "best_qid": best_qid,
            "type_signal": type_signal,
            "qt_bonus": qt_bonus,
            "term_match_signal": term_match_signal,
            "entailment_signal": entailment_signal,
            "temporal_signal": temporal_signal,
            "geo_signal": geo_signal,
            "bidirectional_signal": bidirectional_signal,
            "evidence_chain": self._build_evidence_chain(
                candidates, graph_facts_used, best.span
            ),
        }

    @staticmethod
    def _compute_final_confidence(
        lat_qids: list[str] | None,
        enable_type_coercion: bool,
        signals: dict[str, Any],
    ) -> float:
        confidence = round(
            min(
                0.30 * signals["extraction_conf"]
                + 0.10 * signals["agreement"]
                + 0.15 * signals["graph_signal"]
                + 0.10 * signals["rank_signal"]
                + signals["qt_bonus"]
                + 0.15 * signals["type_signal"]
                + 0.10 * signals["term_match_signal"]
                + 0.10 * signals["entailment_signal"]
                + 0.05 * signals["temporal_signal"]
                + 0.05 * signals["geo_signal"]
                + 0.05 * signals["frequency_signal"]
                + 0.05 * signals["bidirectional_signal"],
                1.0,
            ),
            3,
        )
        if (
            lat_qids
            and enable_type_coercion
            and signals["type_signal"] == 0.0
            and signals["best_qid"] is not None
        ):
            confidence = round(confidence * 0.3, 3)
        return confidence

    @staticmethod
    def _build_score_answer(
        best: AnswerCandidate,
        candidates: list[AnswerCandidate],
        signals: dict[str, Any],
        confidence: float,
    ) -> FinalAnswer:
        return FinalAnswer(
            answer=best.span,
            confidence=confidence,
            source=best.source,
            url=best.url,
            supporting_passages=[c.passage[:200] for c in candidates[:3]],
            graph_facts=signals["graph_facts_used"][:5],
            confidence_breakdown={
                "extraction_model": round(signals["extraction_conf"], 3),
                "span_agreement": round(signals["agreement"], 3),
                "graph_corroboration": signals["graph_signal"],
                "passage_rank_signal": round(signals["rank_signal"], 3),
                "question_type_bonus": signals["qt_bonus"],
                "type_coercion": signals["type_signal"],
                "term_match": round(signals["term_match_signal"], 3),
                "textual_entailment": round(signals["entailment_signal"], 3),
                "temporal_consistency": signals["temporal_signal"],
                "geospatial_consistency": signals["geo_signal"],
                "frequency_signal": round(signals["frequency_signal"], 3),
                "bidirectional_signal": round(signals["bidirectional_signal"], 3),
            },
            evidence_chain=signals["evidence_chain"],
        )

    def score(  # pylint: disable=too-many-arguments
        self,
        candidates: list[AnswerCandidate],
        graph_results: list[GraphResult],
        question_word_type: str | None = None,
        lat_qids: list[str] | None = None,
        *,
        question: str = "",
        ranked_passages: list[RankedPassage] | None = None,
        enable_question_type_bonus: bool = True,
        enable_type_coercion: bool = True,
        enable_term_match: bool = True,
        enable_consistency: bool = True,
        enable_entailment: bool = True,
        enable_answer_merging: bool = True,
        bidirectional_signal: float = 0.0,
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
        signals = self._compute_all_signals(
            best,
            candidates,
            graph_results,
            question_word_type,
            lat_qids or [],
            question,
            ranked_passages,
            enable_type_coercion,
            enable_answer_merging,
            enable_question_type_bonus,
            enable_term_match,
            enable_consistency,
            enable_entailment,
            bidirectional_signal,
        )
        confidence = self._compute_final_confidence(
            lat_qids, enable_type_coercion, signals
        )
        if (
            self.confidence_threshold is not None
            and confidence < self.confidence_threshold
        ):
            logger.debug(
                "Confidence %.3f below threshold %.3f — abstaining",
                confidence,
                self.confidence_threshold,
            )
            return FinalAnswer(
                answer="I don't know",
                confidence=confidence,
                source="",
                url="",
                confidence_breakdown={
                    "reason": "below_threshold",
                    "threshold": self.confidence_threshold,
                },
            )
        return self._build_score_answer(best, candidates, signals, confidence)
