import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from speech_to_text.models import TranscriptionResult
from speech_to_text import ui
from speech_to_text.ui import STYLE_KEYWORD, style_provider, write_transcription_file


class UiTests(unittest.TestCase):
    def test_write_transcription_file_keeps_files_with_same_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            wav_result = TranscriptionResult(
                file_name="clip.wav",
                date="2026-01-01 00:00:00",
                transcription_text="hola",
                model_name="fake-model",
                provider_name="fake",
            )
            mp3_result = TranscriptionResult(
                file_name="clip.mp3",
                date="2026-01-01 00:00:00",
                transcription_text="mundo",
                model_name="fake-model",
                provider_name="fake",
            )

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                write_transcription_file(wav_result, output_dir)
                write_transcription_file(mp3_result, output_dir)

            self.assertTrue((output_dir / "msg" / "clip.wav.txt").exists())
            self.assertTrue((output_dir / "msg" / "clip.mp3.txt").exists())

    def test_write_transcription_file_preserves_line_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = TranscriptionResult(
                file_name="talk.mp3",
                date="2026-01-01 00:00:00",
                transcription_text="primer párrafo\n\nsegundo párrafo",
                model_name="fake-model",
                provider_name="fake",
            )

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                write_transcription_file(result, output_dir)

            content = (output_dir / "msg" / "talk.mp3.txt").read_text(encoding="utf-8")
            self.assertIn("primer párrafo\n\nsegundo párrafo\n", content)

    def test_write_transcription_file_wraps_long_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = TranscriptionResult(
                file_name="long.mp3",
                date="2026-01-01 00:00:00",
                transcription_text="palabra " * 40,
                model_name="fake-model",
                provider_name="fake",
            )

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                write_transcription_file(result, output_dir)

            content = (output_dir / "msg" / "long.mp3.txt").read_text(encoding="utf-8")
            body = content.split("-" * 85 + "\n")[1]
            self.assertTrue(all(len(line) <= 80 for line in body.splitlines()))

    def _write(self, output_dir: Path, **fields) -> Path:
        result = TranscriptionResult(
            file_name="nota.m4a",
            date="2026-01-01 00:00:00",
            transcription_text=fields.pop("text", ""),
            model_name="fake-model",
            provider_name="fake",
            **fields,
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            write_transcription_file(result, output_dir)
        return output_dir / "msg"

    def test_never_overwrites_an_existing_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            msg_dir = self._write(Path(temp_dir), text="primera grabación")
            self._write(Path(temp_dir), text="segunda grabación")

            self.assertIn("primera grabación", (msg_dir / "nota.m4a.txt").read_text(encoding="utf-8"))
            self.assertIn("segunda grabación", (msg_dir / "nota.m4a.1.txt").read_text(encoding="utf-8"))

    def test_a_retry_replaces_the_previous_error_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            msg_dir = self._write(Path(temp_dir), error="Error: 503")
            self._write(Path(temp_dir), text="ahora sí")

            self.assertEqual([p.name for p in msg_dir.iterdir()], ["nota.m4a.txt"])
            content = (msg_dir / "nota.m4a.txt").read_text(encoding="utf-8")
            self.assertIn("ahora sí", content)
            self.assertNotIn("ERROR DE", content)

    def test_conversion_errors_are_replaced_on_retry_too(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            msg_dir = self._write(Path(temp_dir), conversion_error="FFmpeg falló")
            self._write(Path(temp_dir), text="convertido")

            self.assertEqual([p.name for p in msg_dir.iterdir()], ["nota.m4a.txt"])


class ProviderStyleTests(unittest.TestCase):
    def test_gemini_and_its_models_use_the_gemini_blue(self) -> None:
        self.assertEqual(style_provider("gemini"), "[bold #4796e3]gemini[/]")
        self.assertEqual(style_provider("gemini-3.8-flash", "gemini"), "[bold #4796e3]gemini-3.8-flash[/]")

    def test_no_other_style_uses_a_provider_color(self) -> None:
        provider_colors = {color.split()[-1] for color in ui.PROVIDER_STYLES.values()}
        other_styles = [value for name, value in vars(ui).items() if name.startswith("STYLE_")]
        other_styles += list(ui.STATE_STYLES.values()) + list(ui.UI_TOKENS.values())
        for style in other_styles:
            self.assertFalse(provider_colors & set(str(style).split()), style)

    def test_an_unknown_provider_falls_back_to_the_keyword_style(self) -> None:
        self.assertEqual(style_provider("otro"), f"[{STYLE_KEYWORD}]otro[/]")


if __name__ == "__main__":
    unittest.main()
