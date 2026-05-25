"""End-to-end integration tests for the WatsonLite pipeline.

These tests exercise the real internal components (BM25Retriever, RRFFusion,
ConfidenceScorer, Cache, WikidataGraph, …) while mocking only the external I/O
boundaries: HTTP calls and ML-model inference.  The goal is to validate that
data flows correctly through the full pipeline without relying on network
access or large downloaded models.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from watson_lite.core.models import FinalAnswer, Passage
from watson_lite.pipeline import WatsonLite

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EIFFEL_PASSAGE = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
    "in Paris, France.  It was designed by Gustave Eiffel and built between "
    "1887 and 1889 as the entrance arch to the 1889 World's Fair."
)

_PARIS_PASSAGE = (
    "Paris is the capital and most populous city of France.  It has an "
    "estimated population of more than 2 million residents in the city proper."
)


def _make_passages(*texts: str) -> list[Passage]:
    return [
        Passage(
            text=t,
            source=f"Article {i}",
            url=f"https://en.wikipedia.org/wiki/Article_{i}",
        )
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Fixture: full pipeline with mocked I/O
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_pipeline():
    """Build a WatsonLite instance with all external I/O mocked.

    - Wikipedia HTTP calls are replaced by a canned passage list.
    - spaCy is mocked to return deterministic NLP output.
    - SentenceTransformer encodes passages as simple unit vectors.
    - CrossEncoder returns fixed scores (first passage wins).
    - The QA transformers pipeline returns a fixed span.
    """
    with (
        patch("watson_lite.pipeline.fetch_wikipedia_passages") as mock_fetch,
        patch("watson_lite.core.nlp.spacy") as mock_spacy,
        patch("watson_lite.retrieval.vector_retriever.SentenceTransformer") as mock_st,
        patch("watson_lite.retrieval.vector_retriever.faiss") as mock_faiss,
        patch("watson_lite.ranking.ranker.CrossEncoder") as mock_ce,
        patch("watson_lite.core.extractor.hf_pipeline") as mock_hf_pipe,
        patch("watson_lite.graph.wikidata.SPARQLWrapper"),
        patch("watson_lite.graph.wikidata.requests.get") as mock_wd_get,
        patch("watson_lite.graph.wikidata.get_cache") as mock_wd_cache,
    ):
        passages = _make_passages(_EIFFEL_PASSAGE, _PARIS_PASSAGE)
        mock_fetch.return_value = passages

        # --- spaCy mock ---
        mock_nlp_model = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.noun_chunks = []

        def _fake_token(text: str) -> MagicMock:
            t = MagicMock()
            t.text = text
            t.dep_ = "ROOT"
            t.pos_ = "VERB"
            t.lemma_ = text.lower()
            t.is_stop = False
            t.is_punct = False
            return t

        mock_doc.__iter__ = MagicMock(return_value=iter([_fake_token("designed")]))
        mock_nlp_model.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp_model

        # --- SentenceTransformer mock ---
        dim = 8
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = dim

        def _fake_encode(texts, **kwargs):  # type: ignore[override]
            n = len(texts) if isinstance(texts, list) else 1
            vecs = np.zeros((n, dim), dtype="float32")
            vecs[:, 0] = 1.0  # unit vector in dim-0
            return vecs

        mock_model.encode.side_effect = _fake_encode
        mock_st.return_value = mock_model

        # --- FAISS mock ---
        mock_index = MagicMock()
        mock_index.ntotal = len(passages)
        # Return both passages, first ranked highest
        mock_index.search.return_value = (
            np.array([[0.9, 0.7]]),
            np.array([[0, 1]]),
        )
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()

        # --- CrossEncoder mock ---
        mock_ce_model = MagicMock()
        mock_ce_model.predict.return_value = [0.95, 0.80]
        mock_ce.return_value = mock_ce_model

        # --- HuggingFace QA pipeline mock ---
        mock_qa = MagicMock()
        mock_qa.return_value = {"answer": "Gustave Eiffel", "score": 0.92}
        mock_hf_pipe.return_value = mock_qa

        # --- Wikidata mock (cache miss, no facts) ---
        mock_wd_cache_obj = MagicMock()
        from watson_lite.core.cache import SENTINEL

        mock_wd_cache_obj.get_or_sentinel.return_value = SENTINEL
        mock_wd_cache.return_value = mock_wd_cache_obj
        mock_wd_resp = MagicMock()
        mock_wd_resp.status_code = 200
        mock_wd_resp.json.return_value = {"search": []}
        mock_wd_get.return_value = mock_wd_resp

        watson = WatsonLite()
        yield watson, mock_fetch, mock_qa


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    def test_answer_returns_final_answer(self, patched_pipeline) -> None:
        watson, _, _ = patched_pipeline
        result = watson.answer("Who designed the Eiffel Tower?", verbose=False)
        assert isinstance(result, FinalAnswer)

    def test_answer_span_is_nonempty(self, patched_pipeline) -> None:
        watson, _, _ = patched_pipeline
        result = watson.answer("Who designed the Eiffel Tower?", verbose=False)
        assert result.answer != ""

    def test_answer_confidence_in_range(self, patched_pipeline) -> None:
        watson, _, _ = patched_pipeline
        result = watson.answer("Who designed the Eiffel Tower?", verbose=False)
        assert 0.0 <= result.confidence <= 1.0

    def test_answer_source_populated(self, patched_pipeline) -> None:
        watson, _, _ = patched_pipeline
        result = watson.answer("Who designed the Eiffel Tower?", verbose=False)
        assert result.source != ""

    def test_answer_url_populated(self, patched_pipeline) -> None:
        watson, _, _ = patched_pipeline
        result = watson.answer("Who designed the Eiffel Tower?", verbose=False)
        assert result.url.startswith("http")

    def test_answer_confidence_breakdown_keys(self, patched_pipeline) -> None:
        watson, _, _ = patched_pipeline
        result = watson.answer("Who designed the Eiffel Tower?", verbose=False)
        for key in (
            "extraction_model",
            "span_agreement",
            "graph_corroboration",
            "passage_rank_signal",
            "question_type_bonus",
        ):
            assert key in result.confidence_breakdown, f"Missing key: {key}"

    def test_answer_populates_diagnostics(self, patched_pipeline) -> None:
        watson, _, _ = patched_pipeline
        result = watson.answer("Who designed the Eiffel Tower?", verbose=False)
        diagnostics = result.diagnostics
        assert diagnostics is not None
        assert diagnostics.total_latency_s >= 0.0
        assert diagnostics.stage_latencies_s["nlp"] >= 0.0
        assert diagnostics.stage_latencies_s["retrieval"] >= 0.0
        assert diagnostics.stage_latencies_s["ranking"] >= 0.0
        assert diagnostics.stage_latencies_s["extraction"] >= 0.0
        assert diagnostics.stage_latencies_s["scoring"] >= 0.0
        assert diagnostics.passages_fetched > 0
        assert diagnostics.passages_reranked > 0
        assert diagnostics.passages_extracted > 0

    def test_no_passages_returns_fallback(self, patched_pipeline) -> None:
        watson, mock_fetch, _ = patched_pipeline
        mock_fetch.return_value = []
        result = watson.answer("Who invented fire?", verbose=False)
        assert result.confidence == 0.0
        assert "retrieve" in result.answer.lower() or result.answer != ""

    def test_reindex_skipped_second_call(self, patched_pipeline) -> None:
        """Calling answer() twice with the same passages must produce the same hash."""
        watson, _, _ = patched_pipeline
        # The fixture already configures mock_fetch to return 2 passages.
        watson.answer("Who designed it?", verbose=False)
        first_hash = watson._last_passage_hash

        watson.answer("Who designed it?", verbose=False)
        assert watson._last_passage_hash == first_hash

    def test_different_passages_triggers_reindex(self, patched_pipeline) -> None:
        watson, mock_fetch, _ = patched_pipeline
        # First call uses the fixture's 2-passage set.
        watson.answer("question one", verbose=False)
        hash_one = watson._last_passage_hash

        # Switch to a different set of passages (still 2, so FAISS mock is valid).
        mock_fetch.return_value = _make_passages(_PARIS_PASSAGE, _EIFFEL_PASSAGE)
        watson.answer("question two", verbose=False)
        hash_two = watson._last_passage_hash

        assert hash_one != hash_two

    def test_question_type_who_bonus_applied(self, patched_pipeline) -> None:
        """'Who' question with a multi-word capitalized answer gets a QT bonus."""
        watson, _, mock_qa = patched_pipeline
        mock_qa.return_value = {"answer": "Gustave Eiffel", "score": 0.85}
        result = watson.answer("Who designed the Eiffel Tower?", verbose=False)
        assert result.confidence_breakdown.get("question_type_bonus", 0) == 0.1

    def test_question_type_when_bonus_applied(self, patched_pipeline) -> None:
        """'When' question with a year answer gets a QT bonus."""
        watson, _, mock_qa = patched_pipeline
        mock_qa.return_value = {"answer": "1889", "score": 0.85}
        result = watson.answer("When was the Eiffel Tower built?", verbose=False)
        assert result.confidence_breakdown.get("question_type_bonus", 0) == 0.1

    def test_empty_question_raises(self, patched_pipeline) -> None:
        watson, _, _ = patched_pipeline
        with pytest.raises(ValueError, match="must not be empty"):
            watson.answer("")
