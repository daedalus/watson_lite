import sys

import pytest
from unittest.mock import MagicMock, patch

from watson_lite.__main__ import main


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
            mock_wl.answer.assert_called_once_with(
                "What is Python?", verbose=True
            )

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
            mock_wl.answer.assert_called_once_with(
                "What is Python?", verbose=True
            )

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
