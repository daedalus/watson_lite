"""Tests for pure Python helpers in the type_coercion module."""
import json
from unittest.mock import MagicMock, patch

import pytest

from watson_lite.core.models import AnswerCandidate
from watson_lite.scoring.type_coercion import (
    _batch_fetch_claims,
    _extract_qids_from_claims,
    resolve_span_to_qid,
    score_type_coercion,
)


def _make_candidate(span: str) -> AnswerCandidate:
    return AnswerCandidate(
        span=span, source="s", url="", passage="", extraction_score=0.9, rank=1
    )


class TestExtractQidsFromClaims:
    def test_empty_claims_returns_empty(self) -> None:
        assert _extract_qids_from_claims({}, "P31") == []

    def test_missing_pid_returns_empty(self) -> None:
        claims: dict = {"P31": []}
        assert _extract_qids_from_claims(claims, "P279") == []

    def test_extracts_qid_from_value_claim(self) -> None:
        claims = {
            "P31": [
                {
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q5"}},
                    }
                }
            ]
        }
        result = _extract_qids_from_claims(claims, "P31")
        assert result == ["Q5"]

    def test_skips_non_value_snaktype(self) -> None:
        claims = {
            "P31": [
                {
                    "mainsnak": {
                        "snaktype": "somevalue",
                        "datavalue": {"value": {"id": "Q5"}},
                    }
                }
            ]
        }
        assert _extract_qids_from_claims(claims, "P31") == []

    def test_skips_non_qid_value(self) -> None:
        claims = {
            "P31": [
                {
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": "not a dict"},
                    }
                }
            ]
        }
        assert _extract_qids_from_claims(claims, "P31") == []

    def test_skips_non_q_prefixed_id(self) -> None:
        claims = {
            "P31": [
                {
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "P31"}},
                    }
                }
            ]
        }
        assert _extract_qids_from_claims(claims, "P31") == []

    def test_extracts_multiple_qids(self) -> None:
        claims = {
            "P31": [
                {
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q5"}},
                    }
                },
                {
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q215627"}},
                    }
                },
            ]
        }
        result = _extract_qids_from_claims(claims, "P31")
        assert set(result) == {"Q5", "Q215627"}


class TestBatchFetchClaims:
    def test_empty_qids_returns_empty(self) -> None:
        assert _batch_fetch_claims([]) == {}

    def test_successful_fetch_returns_claims(self) -> None:
        api_response = {
            "entities": {
                "Q90": {
                    "claims": {
                        "P31": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {"value": {"id": "Q515"}},
                                }
                            }
                        ],
                        "P279": [],
                    }
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response

        with patch(
            "watson_lite.scoring.type_coercion.requests.get", return_value=mock_resp
        ):
            result = _batch_fetch_claims(["Q90"])

        assert "Q90" in result
        assert result["Q90"]["P31"] == ["Q515"]

    def test_non_200_status_returns_empty(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch(
            "watson_lite.scoring.type_coercion.requests.get", return_value=mock_resp
        ):
            result = _batch_fetch_claims(["Q90"])

        assert result == {}

    def test_exception_returns_empty(self) -> None:
        with patch(
            "watson_lite.scoring.type_coercion.requests.get",
            side_effect=ConnectionError("timeout"),
        ):
            result = _batch_fetch_claims(["Q90"])

        assert result == {}


class TestResolveSpanToQid:
    def test_cache_hit_returns_cached_qid(self) -> None:
        mock_cache = MagicMock()
        mock_cache.get_or_sentinel.return_value = "Q90"

        with (
            patch(
                "watson_lite.scoring.type_coercion.get_cache", return_value=mock_cache
            ),
            patch(
                "watson_lite.scoring.type_coercion.is_cache_miss", return_value=False
            ),
        ):
            result = resolve_span_to_qid("Paris")

        assert result == "Q90"

    def test_cache_hit_none_returns_none(self) -> None:
        mock_cache = MagicMock()
        mock_cache.get_or_sentinel.return_value = None

        with (
            patch(
                "watson_lite.scoring.type_coercion.get_cache", return_value=mock_cache
            ),
            patch(
                "watson_lite.scoring.type_coercion.is_cache_miss", return_value=False
            ),
        ):
            result = resolve_span_to_qid("Unknown entity")

        assert result is None

    def test_successful_api_fetch_returns_qid(self) -> None:
        api_response = {"search": [{"id": "Q90", "label": "Paris"}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response
        mock_cache = MagicMock()
        sentinel = object()
        mock_cache.get_or_sentinel.return_value = sentinel

        with (
            patch(
                "watson_lite.scoring.type_coercion.get_cache", return_value=mock_cache
            ),
            patch(
                "watson_lite.scoring.type_coercion.is_cache_miss", return_value=True
            ),
            patch(
                "watson_lite.scoring.type_coercion.requests.get",
                return_value=mock_resp,
            ),
        ):
            result = resolve_span_to_qid("Paris")

        assert result == "Q90"
        mock_cache.set.assert_called_once_with(
            f"tc:entity:paris", "Q90"
        )

    def test_api_non_200_returns_none(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_cache = MagicMock()
        sentinel = object()
        mock_cache.get_or_sentinel.return_value = sentinel

        with (
            patch(
                "watson_lite.scoring.type_coercion.get_cache", return_value=mock_cache
            ),
            patch(
                "watson_lite.scoring.type_coercion.is_cache_miss", return_value=True
            ),
            patch(
                "watson_lite.scoring.type_coercion.requests.get",
                return_value=mock_resp,
            ),
        ):
            result = resolve_span_to_qid("Paris")

        assert result is None

    def test_api_empty_search_returns_none(self) -> None:
        api_response: dict = {"search": []}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response
        mock_cache = MagicMock()
        sentinel = object()
        mock_cache.get_or_sentinel.return_value = sentinel

        with (
            patch(
                "watson_lite.scoring.type_coercion.get_cache", return_value=mock_cache
            ),
            patch(
                "watson_lite.scoring.type_coercion.is_cache_miss", return_value=True
            ),
            patch(
                "watson_lite.scoring.type_coercion.requests.get",
                return_value=mock_resp,
            ),
        ):
            result = resolve_span_to_qid("NoSuchEntity")

        assert result is None

    def test_api_exception_returns_none(self) -> None:
        mock_cache = MagicMock()
        sentinel = object()
        mock_cache.get_or_sentinel.return_value = sentinel

        with (
            patch(
                "watson_lite.scoring.type_coercion.get_cache", return_value=mock_cache
            ),
            patch(
                "watson_lite.scoring.type_coercion.is_cache_miss", return_value=True
            ),
            patch(
                "watson_lite.scoring.type_coercion.requests.get",
                side_effect=ConnectionError("timeout"),
            ),
        ):
            result = resolve_span_to_qid("Paris")

        assert result is None


class TestFetchTypeHierarchy:
    def test_in_memory_cache_hit_returns_cached_set(self) -> None:
        from watson_lite.scoring import type_coercion as tc_mod

        original = dict(tc_mod._type_cache)
        try:
            tc_mod._type_cache["Q_TEST_HIT"] = {"Q_TEST_HIT", "Q5"}
            from watson_lite.scoring.type_coercion import _fetch_type_hierarchy

            result = _fetch_type_hierarchy("Q_TEST_HIT")
            assert result == {"Q_TEST_HIT", "Q5"}
        finally:
            tc_mod._type_cache.clear()
            tc_mod._type_cache.update(original)

    def test_disk_cache_hit_populates_memory_cache(self) -> None:
        from watson_lite.scoring import type_coercion as tc_mod
        from watson_lite.scoring.type_coercion import _fetch_type_hierarchy

        original = dict(tc_mod._type_cache)
        try:
            tc_mod._type_cache.pop("Q_DISK_HIT", None)
            mock_cache = MagicMock()
            mock_cache.get_or_sentinel.return_value = ["Q_DISK_HIT", "Q5"]

            with (
                patch(
                    "watson_lite.scoring.type_coercion.get_cache",
                    return_value=mock_cache,
                ),
                patch(
                    "watson_lite.scoring.type_coercion.is_cache_miss",
                    return_value=False,
                ),
            ):
                result = _fetch_type_hierarchy("Q_DISK_HIT")

            assert "Q5" in result
            assert "Q_DISK_HIT" in tc_mod._type_cache
        finally:
            tc_mod._type_cache.clear()
            tc_mod._type_cache.update(original)

    def test_fresh_fetch_traverses_hierarchy(self) -> None:
        from watson_lite.scoring import type_coercion as tc_mod
        from watson_lite.scoring.type_coercion import _fetch_type_hierarchy

        original = dict(tc_mod._type_cache)
        try:
            tc_mod._type_cache.pop("Q_FRESH", None)
            mock_cache = MagicMock()
            sentinel = object()
            mock_cache.get_or_sentinel.return_value = sentinel

            with (
                patch(
                    "watson_lite.scoring.type_coercion.get_cache",
                    return_value=mock_cache,
                ),
                patch(
                    "watson_lite.scoring.type_coercion.is_cache_miss",
                    return_value=True,
                ),
                patch(
                    "watson_lite.scoring.type_coercion._batch_fetch_claims",
                    return_value={
                        "Q_FRESH": {"P31": ["Q5"], "P279": []}
                    },
                ),
            ):
                result = _fetch_type_hierarchy("Q_FRESH", max_depth=1)

            assert "Q_FRESH" in result
            assert "Q5" in result
        finally:
            tc_mod._type_cache.clear()
            tc_mod._type_cache.update(original)


class TestScoreTypeCoercion:
    def test_empty_candidates_returns_zero(self) -> None:
        assert score_type_coercion([], ["Q5"]) == 0.0

    def test_empty_lat_qids_returns_zero(self) -> None:
        assert score_type_coercion([_make_candidate("Paris")], []) == 0.0

    def test_no_resolved_qid_returns_zero(self) -> None:
        with patch(
            "watson_lite.scoring.type_coercion.resolve_span_to_qid",
            return_value=None,
        ):
            result = score_type_coercion([_make_candidate("Paris")], ["Q5"])
        assert result == 0.0

    def test_exact_qid_match_returns_one(self) -> None:
        with (
            patch(
                "watson_lite.scoring.type_coercion.resolve_span_to_qid",
                return_value="Q5",
            ),
            patch(
                "watson_lite.scoring.type_coercion._fetch_type_hierarchy",
                return_value={"Q5"},
            ),
        ):
            result = score_type_coercion([_make_candidate("A Person")], ["Q5"])
        assert result == 1.0

    def test_ancestor_qid_match_returns_half(self) -> None:
        with (
            patch(
                "watson_lite.scoring.type_coercion.resolve_span_to_qid",
                return_value="Q215627",
            ),
            patch(
                "watson_lite.scoring.type_coercion._fetch_type_hierarchy",
                return_value={"Q215627", "Q5"},
            ),
        ):
            result = score_type_coercion([_make_candidate("A Person")], ["Q5"])
        # Q215627 != "Q5" so we get 0.5
        assert result == 0.5

    def test_no_ancestor_match_returns_zero(self) -> None:
        with (
            patch(
                "watson_lite.scoring.type_coercion.resolve_span_to_qid",
                return_value="Q350",
            ),
            patch(
                "watson_lite.scoring.type_coercion._fetch_type_hierarchy",
                return_value={"Q350"},
            ),
        ):
            result = score_type_coercion([_make_candidate("A City")], ["Q5"])
        assert result == 0.0

