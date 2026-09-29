import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from speech_to_text.audio import AudioValidation
from speech_to_text.models import TranscriptionResult
from speech_to_text.providers import TranscriptionProvider
from speech_to_text.transcriber import Transcriber

PASS_THROUGH = AudioValidation(is_valid=True, error_message=None, needs_conversion=False)


class FakeProvider(TranscriptionProvider):
    name = "fake"

    def __init__(self) -> None:
        self.transcribed_files: List[str] = []
        self.transcribed_paths: List[Path] = []
        self.prompts: List[str] = []

    def initialize(self) -> None:
        return None

    def transcribe(
        self,
        audio_path: Path,
        original_filename: str,
        prompt: str = "",
    ) -> TranscriptionResult:
        self.transcribed_files.append(original_filename)
        self.transcribed_paths.append(audio_path)
        self.prompts.append(prompt)
        return TranscriptionResult(
            file_name=original_filename,
            date="2026-01-01 00:00:00",
            transcription_text=f"texto-{audio_path.name}",
            model_name=self.current_model(),
            provider_name=self.name,
            transcription_time=1.5,
        )

    def available_models(self) -> List[str]:
        return [self.current_model()]

    def max_file_size_mb(self) -> float:
        return 10.0

    def current_model(self) -> str:
        return "fake-model"


class TranscriberTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.input_dir = base / "in"
        self.output_dir = base / "out"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.written: List[TranscriptionResult] = []

    def _run(self, provider: Optional[TranscriptionProvider] = None, prompt: str = "", **kwargs) -> int:
        kwargs.setdefault("ffmpeg_checker", lambda: False)
        kwargs.setdefault("audio_validator", lambda *_: PASS_THROUGH)
        kwargs.setdefault("result_writer", lambda result, _: self.written.append(result))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            transcriber = Transcriber(
                provider=provider or FakeProvider(),
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                **kwargs,
            )
            return transcriber.process_files(prompt=prompt)

    def _audio(self, name: str, content: bytes = b"audio") -> Path:
        path = self.input_dir / name
        path.write_bytes(content)
        return path

    def test_records_conversion_errors_without_calling_provider(self) -> None:
        provider = FakeProvider()
        original = self._audio("clip.wav")

        failed = self._run(
            provider,
            ffmpeg_checker=lambda: True,
            audio_validator=lambda *_: AudioValidation(True, None, needs_conversion=True),
            audio_converter=lambda *_: ([], "fallo de conversion"),
            now_fn=lambda: datetime(2026, 1, 2, 3, 4, 5),
        )

        self.assertEqual(provider.transcribed_files, [])
        self.assertEqual(len(self.written), 1)
        self.assertEqual(self.written[0].conversion_error, "fallo de conversion")
        self.assertEqual(self.written[0].model_name, "fake-model")
        self.assertEqual(failed, 1)
        self.assertTrue(original.exists())  # un fallo no toca el original

    def test_transcribes_and_archives_valid_audio(self) -> None:
        provider = FakeProvider()
        self._audio("voice.mp3")

        failed = self._run(provider, prompt="hola")

        self.assertEqual(failed, 0)
        self.assertTrue((self.output_dir / "audio" / "voice.mp3").exists())
        self.assertFalse((self.input_dir / "voice.mp3").exists())
        self.assertEqual(provider.transcribed_files, ["voice.mp3"])
        self.assertEqual(provider.prompts, ["hola"])
        self.assertIsNone(self.written[0].error)

    def test_converted_audio_is_sent_from_a_temp_dir_and_the_original_is_archived(self) -> None:
        provider = FakeProvider()
        self._audio("clip.wav", b"wav-audio")
        converted: List[Path] = []

        def converter(path: Path, output_dir: Path, segment_sec):
            output_dir.mkdir(parents=True)
            part = output_dir / "part_000.mp3"
            part.write_bytes(b"mp3-audio")
            converted.append(part)
            return [part], None

        self._run(
            provider,
            ffmpeg_checker=lambda: True,
            audio_validator=lambda *_: AudioValidation(True, None, needs_conversion=True),
            audio_converter=converter,
        )

        self.assertEqual(provider.transcribed_files, ["clip.wav"])  # el resultado usa el nombre original
        self.assertEqual(provider.transcribed_paths, converted)  # pero se envía lo convertido
        self.assertEqual((self.output_dir / "audio" / "clip.wav").read_bytes(), b"wav-audio")
        self.assertEqual(list(self.input_dir.iterdir()), [])
        self.assertFalse(converted[0].exists())  # el directorio temporal se limpia
        self.assertEqual(self.written[0].file_name, "clip.wav")

    def test_long_audio_is_split_and_the_parts_are_joined_in_order(self) -> None:
        self._audio("meeting.m4a")
        received = {}

        def converter(path: Path, output_dir: Path, segment_sec):
            received["segment_sec"] = segment_sec
            output_dir.mkdir(parents=True)
            parts = [output_dir / f"part_{i:03d}.mp3" for i in range(3)]
            for part in parts:
                part.write_bytes(b"x")
            return parts, None

        class LimitedProvider(FakeProvider):
            def max_duration_sec(self):
                return 1000.0

        provider = LimitedProvider()
        self._run(
            provider,
            ffmpeg_checker=lambda: True,
            audio_validator=lambda *_: AudioValidation(True, None, True, duration_sec=2500.0),
            audio_converter=converter,
        )

        self.assertEqual(received["segment_sec"], 900.0)  # 90 % del máximo por petición
        self.assertEqual([p.name for p in provider.transcribed_paths], ["part_000.mp3", "part_001.mp3", "part_002.mp3"])
        self.assertEqual(len(self.written), 1)
        self.assertEqual(self.written[0].transcription_text, "texto-part_000.mp3\ntexto-part_001.mp3\ntexto-part_002.mp3")
        self.assertEqual(self.written[0].transcription_time, 4.5)
        self.assertTrue((self.output_dir / "audio" / "meeting.m4a").exists())

    def test_short_audio_is_not_split(self) -> None:
        received = {}

        def converter(path: Path, output_dir: Path, segment_sec):
            received["segment_sec"] = segment_sec
            output_dir.mkdir(parents=True)
            part = output_dir / "part_000.mp3"
            part.write_bytes(b"x")
            return [part], None

        class LimitedProvider(FakeProvider):
            def max_duration_sec(self):
                return 1000.0

        self._audio("short.wav")
        self._run(
            LimitedProvider(),
            ffmpeg_checker=lambda: True,
            audio_validator=lambda *_: AudioValidation(True, None, True, duration_sec=600.0),
            audio_converter=converter,
        )

        self.assertIsNone(received["segment_sec"])

    def test_an_error_in_any_part_fails_the_file_and_keeps_the_original(self) -> None:
        self._audio("meeting.m4a")

        class FlakyProvider(FakeProvider):
            def transcribe(self, audio_path, original_filename, prompt=""):
                result = super().transcribe(audio_path, original_filename, prompt)
                if audio_path.name == "part_001.mp3":
                    result.error = "Error: 503"
                return result

        provider = FlakyProvider()

        def converter(path: Path, output_dir: Path, segment_sec):
            output_dir.mkdir(parents=True)
            parts = [output_dir / f"part_{i:03d}.mp3" for i in range(3)]
            for part in parts:
                part.write_bytes(b"x")
            return parts, None

        failed = self._run(
            provider,
            ffmpeg_checker=lambda: True,
            audio_validator=lambda *_: AudioValidation(True, None, True),
            audio_converter=converter,
        )

        self.assertEqual(failed, 1)
        self.assertEqual(self.written[0].error, "Tramo 2/3: Error: 503")
        self.assertEqual([p.name for p in provider.transcribed_paths], ["part_000.mp3", "part_001.mp3"])
        self.assertTrue((self.input_dir / "meeting.m4a").exists())
        self.assertFalse((self.output_dir / "audio").exists())

    def test_files_without_audio_are_ignored_and_left_untouched(self) -> None:
        notes = self._audio("notas.txt", b"no es audio")
        provider = FakeProvider()

        failed = self._run(
            provider,
            ffmpeg_checker=lambda: True,
            audio_validator=lambda *_: AudioValidation(False, "sin audio", False, is_audio=False),
        )

        self.assertEqual(failed, 0)  # no es un fallo del usuario
        self.assertEqual(provider.transcribed_files, [])
        self.assertEqual(self.written, [])
        self.assertTrue(notes.exists())

    def test_without_ffmpeg_only_formats_gemini_takes_directly_are_collected(self) -> None:
        provider = FakeProvider()
        self._audio("voice.mp3")
        self._audio("old.wma")
        self._audio("notas.txt")

        self._run(provider)

        self.assertEqual(provider.transcribed_files, ["voice.mp3"])
        self.assertTrue((self.input_dir / "old.wma").exists())

    def test_with_ffmpeg_every_visible_file_is_inspected(self) -> None:
        inspected: List[str] = []

        def validator(path, *_):
            inspected.append(path.name)
            return AudioValidation(False, "sin audio", False, is_audio=False)

        self._audio("clip.mkv")
        self._audio("sin_extension")
        self._audio(".oculto")

        self._run(ffmpeg_checker=lambda: True, audio_validator=validator)

        self.assertEqual(inspected, ["clip.mkv", "sin_extension"])

    def test_records_move_failures_as_postprocess_warning(self) -> None:
        provider = FakeProvider()
        self._audio("voice.mp3")

        self._run(provider, file_mover=lambda *_: (_ for _ in ()).throw(OSError("permiso denegado")))

        self.assertEqual(provider.transcribed_files, ["voice.mp3"])
        self.assertEqual(len(self.written), 1)
        self.assertIsNone(self.written[0].error)
        self.assertIn("permiso denegado", self.written[0].postprocess_warning or "")

    def test_never_overwrites_an_already_archived_audio(self) -> None:
        (self.output_dir / "audio").mkdir()
        (self.output_dir / "audio" / "voice.mp3").write_bytes(b"grabacion vieja")
        self._audio("voice.mp3", b"grabacion nueva")

        self._run(result_writer=lambda *_: None)

        self.assertEqual((self.output_dir / "audio" / "voice.mp3").read_bytes(), b"grabacion vieja")
        self.assertEqual((self.output_dir / "audio" / "voice.1.mp3").read_bytes(), b"grabacion nueva")
        self.assertFalse((self.input_dir / "voice.mp3").exists())

    def test_passes_provider_limits_to_the_validator(self) -> None:
        received: List[tuple] = []

        class LimitedProvider(FakeProvider):
            def max_duration_sec(self):
                return 600.0

        def audio_validator(*args) -> AudioValidation:
            received.append(args)
            return PASS_THROUGH

        self._audio("voice.mp3")

        self._run(LimitedProvider(), ffmpeg_checker=lambda: True, audio_validator=audio_validator)

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

        invalid = AudioValidation(False, "demasiado corto", False)
        self._audio("voice.mp3")

        self.assertEqual(self._run(FailingProvider()), 1)
        self.assertEqual(self._run(FakeProvider(), audio_validator=lambda *_: invalid), 1)

        (self.input_dir / "voice.mp3").unlink(missing_ok=True)
        self._audio("voice2.mp3")
        self.assertEqual(self._run(FakeProvider()), 0)

    def test_empty_input_directory_is_not_an_error(self) -> None:
        self.assertEqual(self._run(), 0)


if __name__ == "__main__":
    unittest.main()
