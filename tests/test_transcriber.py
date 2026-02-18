import tempfile
import unittest
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
        response_format: str = "text",
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

            moved_audio = output_dir / "audio" / "voice.mp3"
            self.assertTrue(moved_audio.exists())

        self.assertEqual(provider.transcribed_files, ["voice.mp3"])
        self.assertEqual(len(written_results), 1)
        self.assertIsNone(written_results[0].error)


if __name__ == "__main__":
    unittest.main()
