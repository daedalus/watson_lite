from watson_lite.core.models import Passage
from watson_lite.ranking.ranker import RRFFusion


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
