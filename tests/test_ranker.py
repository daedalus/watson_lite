import pytest
from unittest.mock import MagicMock, patch

from watson_lite.core.models import Passage, RankedPassage
from watson_lite.ranking.ranker import (
    CrossEncoderReranker,
    Ranker,
    RRFFusion,
)


class TestRRFFusion:
    def setup_method(self) -> None:
        self.rrf = RRFFusion()

    def test_empty_lists(self) -> None:
        result = self.rrf.fuse([[], []])
        assert result == []

    def test_single_list(self) -> None:
        p1 = Passage(text="alpha", source="s", url="u")
        p2 = Passage(text="beta", source="s", url="u")
        result = self.rrf.fuse([[p1, p2]], k=60)
        assert len(result) == 2
        assert result[0].text == "alpha"

    def test_two_lists(self) -> None:
        p1 = Passage(text="common", source="s", url="u")
        p2 = Passage(text="unique", source="s", url="u")
        result = self.rrf.fuse([[p1], [p1, p2]], k=60)
        assert len(result) == 2
        assert result[0].text == "common"

    def test_ranking_order(self) -> None:
        p1 = Passage(text="lower", source="s", url="u")
        p2 = Passage(text="higher", source="s", url="u")
        result = self.rrf.fuse([[p2, p1], [p2, p1]], k=1)
        assert result[0].text == "higher"

    def test_scores_assigned(self) -> None:
        p1 = Passage(text="doc1", source="s", url="u")
        result = self.rrf.fuse([[p1]])
        assert result[0].score > 0.0
        assert result[0].rank == 1


class TestCrossEncoderReranker:
    def setup_method(self) -> None:
        self.ce_patcher = patch("watson_lite.ranking.ranker.CrossEncoder")
        self.mock_ce_cls = self.ce_patcher.start()
        self.mock_model = MagicMock()
        self.mock_ce_cls.return_value = self.mock_model
        self.reranker = CrossEncoderReranker()

    def teardown_method(self) -> None:
        self.ce_patcher.stop()

    def test_init_loads_model(self) -> None:
        self.mock_ce_cls.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L6-v2", max_length=512
        )

    def test_rerank_empty(self) -> None:
        result = self.reranker.rerank("query", [])
        assert result == []

    def test_rerank_with_results(self) -> None:
        passages = [
            Passage(
                text="Paris is capital of France.",
                source="Wiki",
                url="http://example.com",
                score=0.8,
            ),
            Passage(
                text="London is capital of UK.",
                source="Wiki",
                url="http://example.com",
                score=0.7,
            ),
        ]
        self.mock_model.predict.return_value = [0.95, 0.85]

        result = self.reranker.rerank("test query", passages, top_k=2)
        assert len(result) == 2
        assert result[0].cross_score == 0.95
        assert result[0].rrf_score == 0.8
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_rerank_top_k_limits(self) -> None:
        passages = [
            Passage(
                text=f"Doc {i}",
                source="S",
                url=f"http://e/{i}",
                score=0.5,
            )
            for i in range(5)
        ]
        self.mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

        result = self.reranker.rerank("query", passages, top_k=3)
        assert len(result) == 3

    def test_rerank_reorders_by_score_desc(self) -> None:
        passages = [
            Passage(
                text="low",
                source="S",
                url="http://e",
                score=0.0,
            ),
            Passage(
                text="high",
                source="S",
                url="http://e",
                score=0.0,
            ),
        ]
        self.mock_model.predict.return_value = [0.1, 0.9]

        result = self.reranker.rerank("query", passages, top_k=2)
        assert result[0].passage.text == "high"


class TestRanker:
    def setup_method(self) -> None:
        self.ce_patcher = patch("watson_lite.ranking.ranker.CrossEncoder")
        self.mock_ce_cls = self.ce_patcher.start()
        self.mock_model = MagicMock()
        self.mock_ce_cls.return_value = self.mock_model
        self.ranker = Ranker()

    def teardown_method(self) -> None:
        self.ce_patcher.stop()

    def test_rank_full_flow(self) -> None:
        bm25_results = [
            Passage(
                text="Paris is the capital of France.",
                source="Wiki",
                url="http://e.com",
            ),
        ]
        vector_results = [
            Passage(
                text="Paris is the capital of France.",
                source="Wiki",
                url="http://e.com",
            ),
            Passage(
                text="London is a big city.",
                source="Wiki",
                url="http://e.com",
            ),
        ]
        self.mock_model.predict.return_value = [0.9, 0.7]

        result = self.ranker.rank(
            "What is the capital of France?",
            bm25_results,
            vector_results,
            top_k=10,
        )
        assert len(result) >= 1
        assert isinstance(result[0], RankedPassage)

    def test_rank_with_empty_bm25(self) -> None:
        self.mock_model.predict.return_value = [0.5]

        result = self.ranker.rank(
            "test", [], [Passage(text="x", source="s", url="u")], top_k=10
        )
        assert len(result) == 1

    def test_rank_with_empty_vector(self) -> None:
        self.mock_model.predict.return_value = [0.5]

        result = self.ranker.rank(
            "test", [Passage(text="x", source="s", url="u")], [], top_k=10
        )
        assert len(result) == 1

    def test_rank_with_both_empty(self) -> None:
        result = self.ranker.rank("test", [], [], top_k=10)
        assert result == []

    def test_rank_without_cross_encoder(self) -> None:
        result = self.ranker.rank(
            "test",
            [Passage(text="x", source="s", url="u")],
            [Passage(text="y", source="s", url="u")],
            top_k=2,
            use_cross_encoder=False,
        )
        assert len(result) == 2
        assert result[0].final_score > 0.0
        self.mock_model.predict.assert_not_called()
