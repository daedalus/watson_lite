import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from watson_lite.__main__ import (
    _build_config,
    _build_parser,
    _emit_answer,
    _parse_datasets,
    _print_text_answer,
    _run_batch_mode,
    _run_single_question,
    _setup_logging,
    main,
)
from watson_lite.core.config import FeatureConfig
from watson_lite.core.models import AnswerDiagnostics, FinalAnswer


class TestMain:
    @staticmethod
    def _fake_answer() -> FinalAnswer:
        return FinalAnswer(
            answer="Python",
            confidence=0.9,
            source="Wiki",
            url="https://example.org",
        )

    def test_main_with_argv(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog", "What", "is", "Python?"]),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            mock_wl_cls.assert_called_once()
            mock_wl.answer.assert_called_once_with("What is Python?", verbose=False)

    def test_main_with_feature_flags(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--no-vector-retrieval",
                    "--no-graph-enrichment",
                    "What",
                    "is",
                    "Python?",
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert isinstance(called_config, FeatureConfig)
            assert called_config.vector_retrieval is False
            assert called_config.graph_enrichment is False
            mock_wl.answer.assert_called_once_with("What is Python?", verbose=False)

    def test_main_with_new_deepqa_flags(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--no-multi-hypothesis",
                    "--no-per-candidate-retrieval",
                    "--no-bidirectional-validation",
                    "--no-iterative-retrieval",
                    "--semantic-nlp",
                    "--max-retrieval-passes",
                    "3",
                    "--iterative-retrieval-threshold",
                    "0.25",
                    "What",
                    "is",
                    "Python?",
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert called_config.multi_hypothesis is False
            assert called_config.per_candidate_retrieval is False
            assert called_config.bidirectional_validation is False
            assert called_config.iterative_retrieval is False
            assert called_config.semantic_nlp is True
            assert called_config.max_retrieval_passes == 3
            assert called_config.iterative_retrieval_threshold == 0.25

    def test_main_with_datasets_flag(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--datasets",
                    "wikipedia,wikibooks",
                    "What",
                    "is",
                    "Python?",
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert isinstance(called_config, FeatureConfig)
            assert called_config.dataset_sources == ("wikipedia", "wikibooks")

    def test_main_with_elasticsearch_flags(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--datasets",
                    "elasticsearch",
                    "--elasticsearch-url",
                    "http://localhost:9200",
                    "--elasticsearch-index",
                    "wiki_passages",
                    "What",
                    "is",
                    "Python?",
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert isinstance(called_config, FeatureConfig)
            assert called_config.dataset_sources == ("elasticsearch",)
            assert called_config.elasticsearch_url == "http://localhost:9200"
            assert called_config.elasticsearch_index == "wiki_passages"

    def test_main_with_huggingface_flags(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--datasets",
                    "huggingface",
                    "--huggingface-dataset",
                    "ag_news",
                    "--huggingface-config",
                    "default",
                    "--huggingface-split",
                    "train",
                    "--huggingface-token",
                    "hf_example",
                    "What",
                    "is",
                    "Python?",
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert isinstance(called_config, FeatureConfig)
            assert called_config.dataset_sources == ("huggingface",)
            assert called_config.huggingface_dataset == "ag_news"
            assert called_config.huggingface_config == "default"
            assert called_config.huggingface_split == "train"
            assert called_config.huggingface_token == "hf_example"

    def test_main_with_new_public_datasets_flag(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--datasets",
                    "wikiquote,wikisource,wikinews,pubmed,arxiv,openlibrary,stackexchange,dbpedia,oeis",
                    "What",
                    "is",
                    "Python?",
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert called_config.dataset_sources == (
                "wikiquote",
                "wikisource",
                "wikinews",
                "pubmed",
                "arxiv",
                "openlibrary",
                "stackexchange",
                "dbpedia",
                "oeis",
            )

    def test_main_benchmark_mode(self) -> None:
        with (
            patch("watson_lite.__main__.run_benchmark_profiles") as mock_run,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--benchmark-dataset",
                    "/tmp/bench.json",
                    "--ablation-sweep",
                    "--regression-check",
                ],
            ),
        ):
            mock_run.return_value = ([], [])
            result = main()

            assert result == 0
            mock_run.assert_called_once()

    def test_main_benchmark_mode_regression_failure(self) -> None:
        with (
            patch("watson_lite.__main__.run_benchmark_profiles") as mock_run,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--benchmark-dataset",
                    "/tmp/bench.json",
                    "--regression-check",
                ],
            ),
        ):
            mock_run.return_value = ([], [{"metric": "f1"}])
            result = main()
            assert result == 1

    def test_main_no_argv_interactive_quit(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog"]),
            patch("builtins.input", side_effect=["quit"]),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0

    def test_main_interactive_answer_then_quit(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog"]),
            patch("builtins.input", side_effect=["What is Python?", "quit"]),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            mock_wl.answer.assert_called_once_with("What is Python?", verbose=False)

    def test_main_interactive_empty_input(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog"]),
            patch("builtins.input", side_effect=["", "quit"]),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            mock_wl.answer.assert_not_called()

    def test_main_interactive_exit_variants(self) -> None:
        for cmd in ("exit", "q"):
            with (
                patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
                patch.object(sys, "argv", ["prog"]),
                patch("builtins.input", side_effect=[cmd]),
            ):
                mock_wl = MagicMock()
                mock_wl.answer.return_value = self._fake_answer()
                mock_wl_cls.return_value = mock_wl

                result = main()
                assert result == 0

    def test_main_keyboard_interrupt(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog"]),
            patch("builtins.input", side_effect=KeyboardInterrupt()),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()
            assert result == 0

    def test_main_eof_error(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog"]),
            patch("builtins.input", side_effect=EOFError()),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()
            assert result == 0

    def test_main_with_minimal_profile(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        answer = FinalAnswer(
            answer="Paris", confidence=0.9, source="Wiki", url="https://e"
        )
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                ["prog", "--profile", "minimal", "What", "is", "Python?"],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = answer
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert called_config.vector_retrieval is False
            assert called_config.graph_enrichment is False
            assert called_config.cross_encoder_reranking is False
            assert called_config.multi_hypothesis is False
            assert called_config.iterative_retrieval is False
            assert "ANSWER" in capsys.readouterr().out

    def test_main_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        answer = FinalAnswer(
            answer="Paris",
            confidence=0.9,
            source="Wiki",
            url="https://e",
            diagnostics=AnswerDiagnostics(total_latency_s=0.1),
        )
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                ["prog", "--output", "json", "What", "is", "Python?"],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = answer
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            output = capsys.readouterr().out
            assert '"answer": "Paris"' in output
            assert '"diagnostics"' in output

    def test_main_clear_cache(self) -> None:
        with (
            patch("watson_lite.__main__.get_cache") as mock_get_cache,
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                ["prog", "--clear-cache", "What", "is", "Python?"],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = FinalAnswer(
                answer="Python",
                confidence=0.9,
                source="Wiki",
                url="https://e",
            )
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            mock_get_cache.return_value.clear.assert_called_once()

    def test_device_flag_defaults_to_cpu(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["hello"])
        assert args.device == -1

    def test_device_flag_cuda(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--device", "0", "hello"])
        assert args.device == 0

    def test_verbose_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["hello"])
        assert args.verbose is False
        args = parser.parse_args(["--verbose", "hello"])
        assert args.verbose is True

    def test_debug_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["hello"])
        assert args.debug is False
        args = parser.parse_args(["--debug", "hello"])
        assert args.debug is True

    def test_logfile_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["hello"])
        assert args.logfile is None
        args = parser.parse_args(["--logfile", "/tmp/test.log", "hello"])
        assert args.logfile == "/tmp/test.log"

    def test_verbose_passed_to_answer(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog", "--verbose", "What", "is", "Python?"]),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()
            assert result == 0
            mock_wl.answer.assert_called_once_with("What is Python?", verbose=True)

    def test_device_passed_to_watsonlite(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys, "argv", ["prog", "--device", "0", "What", "is", "Python?"]
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()
            assert result == 0
            mock_wl_cls.assert_called_once_with(
                config=mock_wl_cls.call_args.kwargs["config"], device=0
            )

    def test_batch_mode(self, tmp_path: Path) -> None:
        questions_file = tmp_path / "questions.txt"
        questions_file.write_text("What is Python?\nWhat is Java?\n", encoding="utf-8")
        output_json = tmp_path / "results.json"

        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--questions-from-file",
                    str(questions_file),
                    "--output-json",
                    str(output_json),
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()

        assert result == 0
        assert mock_wl.answer.call_count == 2
        assert output_json.exists()

    def test_batch_mode_without_output_json(self, tmp_path: Path) -> None:
        questions_file = tmp_path / "questions.txt"
        questions_file.write_text("What is Python?\n", encoding="utf-8")

        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                ["prog", "--questions-from-file", str(questions_file)],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()

        assert result == 0
        mock_wl.answer.assert_called_once()

    def test_single_question_with_output_json(self, tmp_path: Path) -> None:
        output_json = tmp_path / "answer.json"

        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                ["prog", "--output-json", str(output_json), "What", "is", "Python?"],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl

            result = main()

        assert result == 0
        assert output_json.exists()

    def test_print_text_answer_with_graph_facts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        answer = FinalAnswer(
            answer="Paris",
            confidence=0.9,
            source="Wiki",
            url="https://e",
            graph_facts=["Paris is in France", "Paris has 2M people"],
        )
        _print_text_answer(answer, show_diagnostics=False)
        out = capsys.readouterr().out
        assert "GRAPH CORROBORATION" in out
        assert "Paris is in France" in out

    def test_print_text_answer_with_diagnostics(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        answer = FinalAnswer(
            answer="Paris",
            confidence=0.9,
            source="Wiki",
            url="https://e",
            diagnostics=AnswerDiagnostics(
                total_latency_s=0.1,
                stage_latencies_s={"nlp": 0.01, "retrieval": 0.05},
                passages_fetched=3,
                passages_reranked=3,
                passages_extracted=2,
                cache_hits=1,
                cache_misses=2,
            ),
        )
        _print_text_answer(answer, show_diagnostics=True)
        out = capsys.readouterr().out
        assert "Diagnostics" in out
        assert "fetched=3" in out
        assert "hits=1" in out
        assert "nlp=" in out

    def test_print_text_answer_show_diagnostics_none_diagnostics(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        answer = FinalAnswer(
            answer="X",
            confidence=0.5,
            source="src",
            url="https://e",
            diagnostics=None,
        )
        _print_text_answer(answer, show_diagnostics=True)
        out = capsys.readouterr().out
        assert "ANSWER" in out
        assert "Diagnostics" not in out

    def test_setup_logging_with_logfile(self, tmp_path: Path) -> None:
        import logging

        logfile = tmp_path / "test.log"
        parser = _build_parser()
        args = parser.parse_args(["--logfile", str(logfile), "hello"])
        # Reset root logger handlers to avoid interference
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            _setup_logging(args)
        finally:
            root.handlers = original_handlers

    def test_setup_logging_with_debug(self) -> None:
        import logging

        parser = _build_parser()
        args = parser.parse_args(["--debug", "hello"])
        root = logging.getLogger()
        original_level = root.level
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            _setup_logging(args)
            assert root.level == logging.DEBUG
        finally:
            root.level = original_level
            root.handlers = original_handlers

    def test_parse_datasets_empty_raises(self) -> None:
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_datasets("  ,  , ")

    def test_exclude_datasets(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--datasets",
                    "wikipedia,wikibooks,pubmed",
                    "--exclude-datasets",
                    "pubmed",
                    "What",
                    "is",
                    "Python?",
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()
            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert "pubmed" not in called_config.dataset_sources
            assert "wikipedia" in called_config.dataset_sources
            assert "wikibooks" in called_config.dataset_sources

    def test_main_with_offline_dataset_dir(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--datasets",
                    "wikipedia_offline",
                    "--offline-dataset-dir",
                    "/tmp/corpora",
                    "What",
                    "is",
                    "Python?",
                ],
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert called_config.dataset_sources == ("wikipedia_offline",)
            assert called_config.offline_dataset_dir == "/tmp/corpora"

    def test_plugins_list_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(sys, "argv", ["prog", "plugins", "list"]):
            result = main()
        assert result == 0
        output = capsys.readouterr().out
        assert "wikipedia" in output
        assert "wikipedia_offline" in output

    def test_plugins_describe_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(sys, "argv", ["prog", "plugins", "describe", "wikipedia"]):
            result = main()
        assert result == 0
        output = capsys.readouterr().out
        assert "name: wikipedia" in output
        assert "mode: online" in output

    def test_plugins_validate_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "plugins",
                "validate",
                "--datasets",
                "wikipedia,wikipedia_offline",
            ],
        ):
            result = main()
        assert result == 0
        assert "All plugins are available." in capsys.readouterr().out

    def test_plugins_validate_command_unknown_dataset(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.object(
            sys,
            "argv",
            ["prog", "plugins", "validate", "--datasets", "unknown_plugin"],
        ):
            result = main()
        assert result == 1
        assert "Unknown plugins" in capsys.readouterr().out
