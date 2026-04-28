import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio import convert_audio, validate_audio


class AudioValidationTests(unittest.TestCase):
    def test_validate_audio_warns_when_near_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.mp3"
            audio_path.write_bytes(b"a" * 980_000)  # ~0.93 MB

            result = validate_audio(audio_path=audio_path, max_size_mb=1.0, ffmpeg_available=False)

            self.assertTrue(result.is_valid)
            self.assertIsNotNone(result.warning_message)
            self.assertFalse(result.needs_conversion)

    def test_validate_audio_requires_ffmpeg_for_non_target_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.wav"
            audio_path.write_bytes(b"a" * 100)

            result = validate_audio(audio_path=audio_path, max_size_mb=1.0, ffmpeg_available=False)

            self.assertFalse(result.is_valid)
            self.assertIn("FFmpeg", result.error_message or "")

    def test_validate_audio_rejects_too_short_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.mp3"
            audio_path.write_bytes(b"a" * 100)

            result = validate_audio(
                audio_path=audio_path,
                max_size_mb=1.0,
                ffmpeg_available=True,
                duration_fn=lambda _: 0.2,
            )

            self.assertFalse(result.is_valid)
            self.assertIn("Audio demasiado corto", result.error_message or "")

    def test_convert_audio_does_not_overwrite_existing_target_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_path = base / "sample.wav"
            existing_target = base / "sample.mp3"
            source_path.write_bytes(b"source")
            existing_target.write_bytes(b"existing")

            def fake_run(command, **_kwargs):
                output_path = Path(command[command.index("128k") + 1])
                output_path.write_bytes(b"converted")

            with patch("audio.subprocess.run", side_effect=fake_run):
                converted_path, error = convert_audio(source_path)

            self.assertIsNone(error)
            self.assertEqual(existing_target.read_bytes(), b"existing")
            self.assertEqual(converted_path, base / "sample.wav.mp3")
            self.assertEqual(converted_path.read_bytes(), b"converted")


if __name__ == "__main__":
    unittest.main()
