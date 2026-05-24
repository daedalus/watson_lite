from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from watson_lite.core.models import Passage
from watson_lite.retrieval.vector_retriever import VectorRetriever


class TestVectorRetriever:
    def setup_method(self) -> None:
        self.st_patcher = patch(
            "watson_lite.retrieval.vector_retriever.SentenceTransformer"
        )
        self.mock_st_cls = self.st_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model.get_sentence_embedding_dimension.return_value = 384
        self.mock_st_cls.return_value = self.mock_model

        self.faiss_patcher = patch("watson_lite.retrieval.vector_retriever.faiss")
        self.mock_faiss = self.faiss_patcher.start()
        self.mock_index = MagicMock()
        self.mock_index.ntotal = 2
        self.mock_faiss.IndexFlatIP.return_value = self.mock_index

        self.retriever = VectorRetriever()

    def teardown_method(self) -> None:
        self.st_patcher.stop()
        self.faiss_patcher.stop()

    def test_init_loads_model(self) -> None:
        self.mock_st_cls.assert_called_once_with("all-MiniLM-L6-v2")
        assert self.retriever.dim == 384
        assert self.retriever.index is None
        assert self.retriever.passages == []

    def test_init_custom_model(self) -> None:
        self.st_patcher.stop()
        self.faiss_patcher.stop()

        with (
            patch(
                "watson_lite.retrieval.vector_retriever.SentenceTransformer"
            ) as mock_cls,
            patch("watson_lite.retrieval.vector_retriever.faiss") as mock_faiss,
        ):
            mock_m = MagicMock()
            mock_m.get_sentence_embedding_dimension.return_value = 768
            mock_cls.return_value = mock_m
            r = VectorRetriever(model_name="custom-model")
            mock_cls.assert_called_once_with("custom-model")
            assert r.dim == 768

    def test_index_passages(self) -> None:
        passages = [
            Passage(
                text="Paris is the capital of France.",
                source="Paris",
                url="http://example.com",
            ),
        ]
        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")

        self.retriever.index_passages(passages)

        assert len(self.retriever.passages) == 1
        self.mock_model.encode.assert_called_once()
        self.mock_faiss.normalize_L2.assert_called()
        self.mock_faiss.IndexFlatIP.assert_called_once_with(384)
        self.mock_index.add.assert_called_once()
        assert self.retriever.index is self.mock_index

    def test_retrieve_no_index(self) -> None:
        assert self.retriever.index is None
        result = self.retriever.retrieve("test")
        assert result == []

    def test_retrieve_with_results(self) -> None:
        passages = [
            Passage(
                text="Paris is the capital of France.",
                source="Paris",
                url="http://example.com",
            ),
            Passage(
                text="London is the capital of the UK.",
                source="London",
                url="http://example.com",
            ),
        ]
        self.retriever.passages = passages
        self.retriever.index = self.mock_index

        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")
        self.mock_index.search.return_value = (
            np.array([[0.95, 0.80]]),
            np.array([[0, 1]]),
        )

        result = self.retriever.retrieve("test query", top_k=10)
        assert len(result) == 2
        assert result[0].text == "Paris is the capital of France."
        assert result[0].score == 0.95
        assert result[0].rank == 1
        assert result[1].score == 0.80
        assert result[1].rank == 2

    def test_retrieve_skips_negative_indices(self) -> None:
        passages = [
            Passage(
                text="Paris is the capital of France.",
                source="Paris",
                url="http://example.com",
            ),
        ]
        self.retriever.passages = passages
        self.retriever.index = self.mock_index

        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")
        self.mock_index.search.return_value = (
            np.array([[0.0]]),
            np.array([[-1]]),
        )

        result = self.retriever.retrieve("test")
        assert result == []

    def test_retrieve_respects_top_k(self) -> None:
        passages = [
            Passage(
                text=f"Doc {i}",
                source="Src",
                url=f"http://example.com/{i}",
            )
            for i in range(5)
        ]
        self.retriever.passages = passages
        self.retriever.index = self.mock_index

        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")
        expected = (
            np.array([[0.9, 0.8, 0.7]]),
            np.array([[0, 1, 2]]),
        )
        self.mock_index.search.return_value = expected

        result = self.retriever.retrieve("test", top_k=3)
        assert len(result) == 3
