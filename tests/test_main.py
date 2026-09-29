import io
import sys
import unittest
from unittest.mock import patch

from speech_to_text.__main__ import build_parser, make_output_safe


class OutputSafetyTests(unittest.TestCase):
    def test_unencodable_characters_do_not_raise_on_a_legacy_codepage(self) -> None:
        raw = io.BytesIO()
        legacy_stdout = io.TextIOWrapper(raw, encoding="cp1252")

        with patch.object(sys, "stdout", legacy_stdout):
            make_output_safe()
            print("guardado: 日本語 reunión.mp3")
            legacy_stdout.flush()

        self.assertEqual(raw.getvalue().decode("cp1252").strip(), "guardado: ??? reunión.mp3")

    def test_streams_without_reconfigure_are_left_alone(self) -> None:
        with patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", io.StringIO()):
            make_output_safe()  # StringIO no tiene reconfigure(): no debe fallar


class ParserTests(unittest.TestCase):
    def test_defaults_to_the_dedicated_model_without_prompt(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.model, "gemini-3.5-transcribe")
        self.assertEqual(args.prompt, "")

    def test_rejects_retired_models(self) -> None:
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--model", "gemini-3.1-flash-lite-preview"])


if __name__ == "__main__":
    unittest.main()
