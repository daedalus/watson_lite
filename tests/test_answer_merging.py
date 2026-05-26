"""Tests for answer merging by Wikidata QID."""

import pytest
from unittest.mock import patch

from watson_lite.core.models import AnswerCandidate
from watson_lite.scoring.answer_merging import merge_candidates_by_qid


def _make_candidate(
    span: str,
    score: float = 0.9,
    rank: int = 1,
    doc_freq: int = 1,
) -> AnswerCandidate:
    return AnswerCandidate(
        span=span,
        source="s",
        url="",
        passage="",
        extraction_score=score,
        rank=rank,
        doc_frequency=doc_freq,
    )


class TestMergeCandidatesByQid:
    def test_single_candidate_returned_unchanged(self) -> None:
        candidates = [_make_candidate("Paris")]
        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            return_value=None,
        ):
            result = merge_candidates_by_qid(candidates)
        assert len(result) == 1
        assert result[0].span == "Paris"

    def test_no_qid_candidates_returned_unchanged(self) -> None:
        candidates = [_make_candidate("Paris"), _make_candidate("London")]
        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            return_value=None,
        ):
            result = merge_candidates_by_qid(candidates)
        assert len(result) == 2

    def test_same_qid_candidates_are_merged(self) -> None:
        candidates = [
            _make_candidate("Paris", score=0.9, rank=1, doc_freq=2),
            _make_candidate("Paris, France", score=0.7, rank=3, doc_freq=1),
        ]
        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            return_value="Q90",
        ):
            result = merge_candidates_by_qid(candidates)

        # Both map to Q90 → merged into one entry
        assert len(result) == 1
        merged = result[0]
        # Canonical is the shortest span
        assert merged.span == "Paris"
        # Best rank across group
        assert merged.rank == 1
        # Max extraction score
        assert merged.extraction_score == pytest.approx(0.9)
        # doc_frequency summed
        assert merged.doc_frequency == 3

    def test_different_qids_not_merged(self) -> None:
        candidates = [
            _make_candidate("Paris"),
            _make_candidate("London"),
        ]
        qid_map = {"Paris": "Q90", "London": "Q84"}
        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            side_effect=lambda span: qid_map.get(span),
        ):
            result = merge_candidates_by_qid(candidates)

        assert len(result) == 2
        spans = {c.span for c in result}
        assert "Paris" in spans
        assert "London" in spans

    def test_merged_result_sorted_by_score(self) -> None:
        candidates = [
            _make_candidate("France", score=0.5, rank=2),
            _make_candidate("Germany", score=0.9, rank=1),
        ]
        with patch(
            "watson_lite.scoring.answer_merging.resolve_span_to_qid",
            return_value=None,
        ):
            result = merge_candidates_by_qid(candidates)

        # ungrouped results are sorted by extraction_score descending
        assert result[0].span == "Germany"
