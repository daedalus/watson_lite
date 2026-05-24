import sys
from unittest.mock import MagicMock, patch

import pytest

from watson_lite.__main__ import main
from watson_lite.core.config import FeatureConfig


class TestMain:
    def test_main_with_argv(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog", "What", "is", "Python?"]),
        ):
            mock_wl = MagicMock()
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            mock_wl_cls.assert_called_once()
            mock_wl.answer.assert_called_once_with("What is Python?", verbose=True)

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
            mock_wl_cls.return_value = mock_wl
            result = main()

            assert result == 0
            called_config = mock_wl_cls.call_args.kwargs["config"]
            assert isinstance(called_config, FeatureConfig)
            assert called_config.vector_retrieval is False
            assert called_config.graph_enrichment is False
            mock_wl.answer.assert_called_once_with("What is Python?", verbose=True)

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
            mock_wl_cls.return_value = mock_wl

            result = main()

            assert result == 0
            mock_wl.answer.assert_called_once_with("What is Python?", verbose=True)

    def test_main_interactive_empty_input(self) -> None:
        with (
            patch("watson_lite.__main__.WatsonLite") as mock_wl_cls,
            patch.object(sys, "argv", ["prog"]),
            patch("builtins.input", side_effect=["", "quit"]),
        ):
            mock_wl = MagicMock()
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
            mock_wl_cls.return_value = mock_wl

            result = main()
            assert result == 0
