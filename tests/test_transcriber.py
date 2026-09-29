import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import List

from audio import AudioValidation
from config import TEMP_SUFFIX
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

    def test_stale_partial_conversions_are_removed_without_touching_user_files(self) -> None:
        provider = FakeProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "in"
            output_dir = base / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            stale_partial = input_dir / f"clip.wav.mp3{TEMP_SUFFIX}"
            user_audio = input_dir / "notes_temp.mp3"
            stale_partial.write_bytes(b"partial")
            user_audio.write_bytes(b"user audio " * 100)

            def run_test() -> None:
                Transcriber(
                    provider=provider,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    ffmpeg_checker=lambda: False,
                    result_writer=lambda *_: None,
                ).process_files()

            self._run_quietly(run_test)

            self.assertFalse(stale_partial.exists())
            self.assertTrue((output_dir / "audio" / "notes_temp.mp3").exists())

        self.assertEqual(provider.transcribed_files, ["notes_temp.mp3"])

    def test_never_overwrites_an_already_archived_audio(self) -> None:
        provider = FakeProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "in"
            output_dir = base / "out"
            (output_dir / "audio").mkdir(parents=True)
            input_dir.mkdir()
            (output_dir / "audio" / "voice.mp3").write_bytes(b"grabacion vieja")
            (input_dir / "voice.mp3").write_bytes(b"grabacion nueva")

            def run_test() -> None:
                Transcriber(
                    provider=provider,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    ffmpeg_checker=lambda: False,
                    audio_validator=lambda *_: AudioValidation(
                        is_valid=True,
                        error_message=None,
                        needs_conversion=False,
                    ),
                    result_writer=lambda *_: None,
                ).process_files()

            self._run_quietly(run_test)

            self.assertEqual((output_dir / "audio" / "voice.mp3").read_bytes(), b"grabacion vieja")
            self.assertEqual((output_dir / "audio" / "voice.1.mp3").read_bytes(), b"grabacion nueva")
            self.assertFalse((input_dir / "voice.mp3").exists())

    def test_passes_provider_duration_limit_to_the_validator(self) -> None:
        received: List[tuple] = []

        class LimitedProvider(FakeProvider):
            def max_duration_sec(self):
                return 600.0

        def audio_validator(*args) -> AudioValidation:
            received.append(args)
            return AudioValidation(is_valid=True, error_message=None, needs_conversion=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "in"
            output_dir = base / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "voice.mp3").write_bytes(b"audio")

            def run_test() -> None:
                Transcriber(
                    provider=LimitedProvider(),
                    input_dir=input_dir,
                    output_dir=output_dir,
                    ffmpeg_checker=lambda: True,
                    audio_validator=audio_validator,
                    result_writer=lambda *_: None,
                ).process_files()

            self._run_quietly(run_test)

        self.assertEqual(received[0][1:], (10.0, True, 600.0))

    def test_process_files_returns_the_number_of_failed_files(self) -> None:
        class FailingProvider(FakeProvider):
            def transcribe(self, audio_path, original_filename, prompt=""):
                return TranscriptionResult(
                    file_name=original_filename,
                    date="2026-01-01 00:00:00",
                    transcription_text="",
                    model_name=self.current_model(),
                    provider_name=self.name,
                    error="Error: 503",
                )

        def process(provider: FakeProvider, invalid: bool = False) -> int:
            with tempfile.TemporaryDirectory() as temp_dir:
                base = Path(temp_dir)
                input_dir = base / "in"
                output_dir = base / "out"
                input_dir.mkdir()
                output_dir.mkdir()
                (input_dir / "voice.mp3").write_bytes(b"audio")
                validation = AudioValidation(
                    is_valid=not invalid,
                    error_message="demasiado corto" if invalid else None,
                    needs_conversion=False,
                )
                transcriber = self._run_quietly(
                    lambda: Transcriber(
                        provider=provider,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        ffmpeg_checker=lambda: False,
                        audio_validator=lambda *_: validation,
                        result_writer=lambda *_: None,
                    )
                )
                return self._run_quietly(transcriber.process_files)

        self.assertEqual(process(FakeProvider()), 0)
        self.assertEqual(process(FailingProvider()), 1)
        self.assertEqual(process(FakeProvider(), invalid=True), 1)


if __name__ == "__main__":
    unittest.main()
