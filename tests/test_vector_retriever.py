import json
import logging
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from watson_lite.core.models import Passage
from watson_lite.retrieval.vector_retriever import EMBED_MODEL, VectorRetriever


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
        self.mock_st_cls.assert_called_once_with(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
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

    def test_init_raises_when_sentence_transformers_missing(self) -> None:
        self.st_patcher.stop()
        self.faiss_patcher.stop()

        with (
            patch("watson_lite.retrieval.vector_retriever.SentenceTransformer", None),
            patch("watson_lite.retrieval.vector_retriever.faiss", MagicMock()),
        ):
            with pytest.raises(ImportError, match="sentence-transformers"):
                VectorRetriever()

    def test_init_raises_when_faiss_missing(self) -> None:
        self.st_patcher.stop()
        self.faiss_patcher.stop()

        with (
            patch(
                "watson_lite.retrieval.vector_retriever.SentenceTransformer",
                MagicMock(),
            ),
            patch("watson_lite.retrieval.vector_retriever.faiss", None),
        ):
            with pytest.raises(ImportError, match="faiss-cpu"):
                VectorRetriever()

    def test_init_raises_when_both_missing(self) -> None:
        self.st_patcher.stop()
        self.faiss_patcher.stop()

        with (
            patch("watson_lite.retrieval.vector_retriever.SentenceTransformer", None),
            patch("watson_lite.retrieval.vector_retriever.faiss", None),
        ):
            with pytest.raises(ImportError, match="sentence-transformers"):
                VectorRetriever()

    def test_save_writes_metadata_with_model_name(self, tmp_path: str) -> None:
        passages = [
            Passage(
                text="Paris is the capital of France.",
                source="Paris",
                url="http://example.com",
            ),  # noqa: E501
        ]
        self.mock_model.encode.return_value = np.array([[0.1] * 384], dtype="float32")
        self.retriever.index_passages(passages)
        self.retriever.save(tmp_path)

        meta_path = os.path.join(tmp_path, "metadata.json")
        assert os.path.isfile(meta_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["embed_model"] == EMBED_MODEL

    def test_load_raises_on_model_mismatch(
        self, tmp_path: str, caplog: pytest.LogCaptureFixture
    ) -> None:  # noqa: E501
        self.st_patcher.stop()
        self.faiss_patcher.stop()

        with (
            patch(
                "watson_lite.retrieval.vector_retriever.SentenceTransformer"
            ) as mock_cls,
            patch("watson_lite.retrieval.vector_retriever.faiss") as mock_faiss,
        ):
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_cls.return_value = mock_model
            mock_idx = MagicMock()
            mock_idx.ntotal = 2
            mock_faiss.read_index.return_value = mock_idx

            # Write metadata claiming a different model
            meta_path = os.path.join(tmp_path, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"embed_model": "some-other-model"}, f)

            # Write dummy passages.json so load can proceed past that
            with open(
                os.path.join(tmp_path, "passages.json"), "w", encoding="utf-8"
            ) as f:
                json.dump([], f)

            with pytest.raises(ValueError, match="some-other-model"):
                VectorRetriever.load(tmp_path, model_name="requested-model")

    def test_load_succeeds_on_model_match(self, tmp_path: str) -> None:
        self.st_patcher.stop()
        self.faiss_patcher.stop()

        with (
            patch(
                "watson_lite.retrieval.vector_retriever.SentenceTransformer"
            ) as mock_cls,
            patch("watson_lite.retrieval.vector_retriever.faiss") as mock_faiss,
        ):
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_cls.return_value = mock_model
            mock_idx = MagicMock()
            mock_idx.ntotal = 2
            mock_faiss.read_index.return_value = mock_idx

            meta_path = os.path.join(tmp_path, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"embed_model": "matching-model"}, f)

            with open(
                os.path.join(tmp_path, "passages.json"), "w", encoding="utf-8"
            ) as f:
                json.dump([], f)

            r = VectorRetriever.load(tmp_path, model_name="matching-model")
            assert r.model_name == "matching-model"

    def test_load_without_metadata_warns_and_proceeds(
        self, tmp_path: str, caplog: pytest.LogCaptureFixture
    ) -> None:  # noqa: E501
        self.st_patcher.stop()
        self.faiss_patcher.stop()

        with (
            patch(
                "watson_lite.retrieval.vector_retriever.SentenceTransformer"
            ) as mock_cls,
            patch("watson_lite.retrieval.vector_retriever.faiss") as mock_faiss,
        ):
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_cls.return_value = mock_model
            mock_idx = MagicMock()
            mock_idx.ntotal = 2
            mock_faiss.read_index.return_value = mock_idx

            with open(
                os.path.join(tmp_path, "passages.json"), "w", encoding="utf-8"
            ) as f:
                json.dump([], f)

            with caplog.at_level(
                logging.WARNING, logger="watson_lite.retrieval.vector_retriever"
            ):
                r = VectorRetriever.load(tmp_path, model_name="fallback-model")
            assert r.model_name == "fallback-model"
            assert "No metadata found" in caplog.text
            assert "fallback-model" in caplog.text
