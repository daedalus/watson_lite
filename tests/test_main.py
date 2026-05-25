import sys
from unittest.mock import MagicMock, patch

import pytest

from watson_lite.__main__ import _build_parser, main
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

    def test_main_with_minimal_profile(self, capsys: pytest.CaptureFixture[str]) -> None:
        answer = FinalAnswer(answer="Paris", confidence=0.9, source="Wiki", url="https://e")
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
            patch.object(
                sys, "argv", ["prog", "--verbose", "What", "is", "Python?"]
            ),
        ):
            mock_wl = MagicMock()
            mock_wl.answer.return_value = self._fake_answer()
            mock_wl_cls.return_value = mock_wl
            result = main()
            assert result == 0
            mock_wl.answer.assert_called_once_with(
                "What is Python?", verbose=True
            )

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
