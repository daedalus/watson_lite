"""Tests for the temporal and geospatial consistency scoring functions."""
import pytest

from watson_lite.core.models import AnswerCandidate, EntityFact, GraphResult
from watson_lite.scoring.consistency import (
    _extract_years,
    score_geospatial_consistency,
    score_temporal_consistency,
)


def _make_candidate(span: str, rank: int = 1) -> AnswerCandidate:
    return AnswerCandidate(
        span=span, source="s", url="", passage="", extraction_score=0.9, rank=rank
    )


def _make_graph(label: str, value: str) -> list[GraphResult]:
    return [
        GraphResult(
            entity_name="ent",
            wikidata_id="Q1",
            facts=[EntityFact(entity="Q1", property_label=label, value=value)],
        )
    ]


class TestExtractYears:
    def test_extracts_four_digit_year(self) -> None:
        assert _extract_years("Built in 1889.") == {"1889"}

    def test_extracts_multiple_years(self) -> None:
        assert _extract_years("From 1900 to 2000.") == {"1900", "2000"}

    def test_no_year(self) -> None:
        assert _extract_years("No date here.") == set()


class TestTemporalConsistency:
    def test_matching_year_returns_one(self) -> None:
        candidates = [_make_candidate("1889")]
        graph = _make_graph("inception", "1889")
        assert score_temporal_consistency(candidates, graph) == 1.0

    def test_non_matching_year_returns_zero(self) -> None:
        candidates = [_make_candidate("1889")]
        graph = _make_graph("inception", "1900")
        assert score_temporal_consistency(candidates, graph) == 0.0

    def test_no_year_in_span_returns_zero(self) -> None:
        candidates = [_make_candidate("Paris")]
        graph = _make_graph("inception", "1889")
        assert score_temporal_consistency(candidates, graph) == 0.0

    def test_non_temporal_label_ignored(self) -> None:
        candidates = [_make_candidate("1889")]
        graph = _make_graph("country", "France")
        assert score_temporal_consistency(candidates, graph) == 0.0

    def test_empty_candidates_returns_zero(self) -> None:
        assert score_temporal_consistency([], _make_graph("inception", "1889")) == 0.0

    def test_empty_graph_returns_zero(self) -> None:
        assert score_temporal_consistency([_make_candidate("1889")], []) == 0.0


class TestGeospatialConsistency:
    def test_exact_match_returns_positive(self) -> None:
        candidates = [_make_candidate("France")]
        graph = _make_graph("country", "France")
        assert score_geospatial_consistency(candidates, graph) > 0.0

    def test_substring_match_returns_positive(self) -> None:
        candidates = [_make_candidate("Paris")]
        graph = _make_graph("location", "Paris, France")
        assert score_geospatial_consistency(candidates, graph) > 0.0

    def test_no_match_returns_zero(self) -> None:
        candidates = [_make_candidate("Germany")]
        graph = _make_graph("country", "France")
        assert score_geospatial_consistency(candidates, graph) == 0.0

    def test_non_geo_label_ignored(self) -> None:
        candidates = [_make_candidate("France")]
        graph = _make_graph("inception", "1889")
        assert score_geospatial_consistency(candidates, graph) == 0.0

    def test_empty_candidates_returns_zero(self) -> None:
        graph = _make_graph("country", "France")
        assert score_geospatial_consistency([], graph) == 0.0

    def test_multiple_matches_capped_at_one(self) -> None:
        candidates = [_make_candidate("Paris")]
        facts = [
            EntityFact(entity="Q1", property_label="capital", value="Paris"),
            EntityFact(entity="Q1", property_label="location", value="Paris"),
            EntityFact(entity="Q1", property_label="headquarters", value="Paris"),
        ]
        graph = [GraphResult(entity_name="ent", wikidata_id="Q1", facts=facts)]
        result = score_geospatial_consistency(candidates, graph)
        assert result == pytest.approx(1.0)

    def test_empty_graph_returns_zero(self) -> None:
        assert score_geospatial_consistency([_make_candidate("France")], []) == 0.0
