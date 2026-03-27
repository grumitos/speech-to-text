import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import List

from audio import AudioValidation
from models import TranscriptionResult
from providers import TranscriptionProvider
from transcriber import Transcriber


class FakeProvider(TranscriptionProvider):
    name = "fake"

    def __init__(self) -> None:
        self.transcribed_files: List[str] = []

    def initialize(self) -> None:
        return None

    def transcribe(
        self,
        audio_path: Path,
        original_filename: str,
        prompt: str = "",
    ) -> TranscriptionResult:
        self.transcribed_files.append(original_filename)
        return TranscriptionResult(
            file_name=original_filename,
            date="2026-01-01 00:00:00",
            transcription_text=f"texto-{audio_path.name}",
            model_name=self.current_model(),
            provider_name=self.name,
        )

    def available_models(self) -> List[str]:
        return [self.current_model()]

    def max_file_size_mb(self) -> float:
        return 10.0

    def current_model(self) -> str:
        return "fake-model"


class TranscriberTests(unittest.TestCase):
    def _run_quietly(self, func):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return func()

    def test_records_conversion_errors_without_calling_provider(self) -> None:
        written_results: List[TranscriptionResult] = []
        provider = FakeProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "in"
            output_dir = base / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "clip.wav").write_bytes(b"abc")

            def run_test() -> None:
                transcriber = Transcriber(
                    provider=provider,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    ffmpeg_checker=lambda: True,
                    audio_validator=lambda *_: AudioValidation(
                        is_valid=True,
                        error_message=None,
                        needs_conversion=True,
                    ),
                    audio_converter=lambda _: (None, "fallo de conversion"),
                    result_writer=lambda result, _: written_results.append(result),
                    now_fn=lambda: datetime(2026, 1, 2, 3, 4, 5),
                )
                transcriber.process_files()

            self._run_quietly(run_test)

        self.assertEqual(provider.transcribed_files, [])
        self.assertEqual(len(written_results), 1)
        self.assertEqual(written_results[0].conversion_error, "fallo de conversion")
        self.assertEqual(written_results[0].model_name, "fake-model")

    def test_transcribes_and_moves_valid_audio(self) -> None:
        written_results: List[TranscriptionResult] = []
        provider = FakeProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "in"
            output_dir = base / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            audio_file = input_dir / "voice.mp3"
            audio_file.write_bytes(b"audio")

            def run_test() -> None:
                transcriber = Transcriber(
                    provider=provider,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    ffmpeg_checker=lambda: False,
                    audio_validator=lambda *_: AudioValidation(
                        is_valid=True,
                        error_message=None,
                        needs_conversion=False,
                    ),
                    result_writer=lambda result, _: written_results.append(result),
                )
                transcriber.process_files(prompt="hola")

            self._run_quietly(run_test)

            moved_audio = output_dir / "audio" / "voice.mp3"
            self.assertTrue(moved_audio.exists())

        self.assertEqual(provider.transcribed_files, ["voice.mp3"])
        self.assertEqual(len(written_results), 1)
        self.assertIsNone(written_results[0].error)

    def test_transcribes_converted_audio_preserving_original_name(self) -> None:
        written_results: List[TranscriptionResult] = []
        provider = FakeProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "in"
            output_dir = base / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            original_audio = input_dir / "clip.wav"
            original_audio.write_bytes(b"wav-audio")

            def audio_validator(file_path: Path, *_args) -> AudioValidation:
                return AudioValidation(
                    is_valid=True,
                    error_message=None,
                    needs_conversion=file_path.suffix.lower() != ".mp3",
                )

            def audio_converter(_path: Path):
                converted_audio = input_dir / "clip.mp3"
                converted_audio.write_bytes(b"mp3-audio")
                return converted_audio, None

            def run_test() -> None:
                transcriber = Transcriber(
                    provider=provider,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    ffmpeg_checker=lambda: True,
                    audio_validator=audio_validator,
                    audio_converter=audio_converter,
                    result_writer=lambda result, _: written_results.append(result),
                )
                transcriber.process_files()

            self._run_quietly(run_test)

            moved_audio = output_dir / "audio" / "clip.mp3"
            self.assertTrue(moved_audio.exists())
            self.assertFalse(original_audio.exists())

        self.assertEqual(provider.transcribed_files, ["clip.wav"])
        self.assertEqual(len(written_results), 1)
        self.assertEqual(written_results[0].file_name, "clip.wav")

    def test_records_move_failures_as_postprocess_warning(self) -> None:
        written_results: List[TranscriptionResult] = []
        provider = FakeProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "in"
            output_dir = base / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            audio_file = input_dir / "voice.mp3"
            audio_file.write_bytes(b"audio")

            def run_test() -> None:
                transcriber = Transcriber(
                    provider=provider,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    ffmpeg_checker=lambda: False,
                    audio_validator=lambda *_: AudioValidation(
                        is_valid=True,
                        error_message=None,
                        needs_conversion=False,
                    ),
                    result_writer=lambda result, _: written_results.append(result),
                    file_mover=lambda *_: (_ for _ in ()).throw(OSError("permiso denegado")),
                )
                transcriber.process_files()

            self._run_quietly(run_test)

        self.assertEqual(provider.transcribed_files, ["voice.mp3"])
        self.assertEqual(len(written_results), 1)
        self.assertIsNone(written_results[0].error)
        self.assertIn("permiso denegado", written_results[0].postprocess_warning or "")


if __name__ == "__main__":
    unittest.main()
