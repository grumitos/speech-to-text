import tempfile
import unittest
from pathlib import Path

from audio import validate_audio


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


if __name__ == "__main__":
    unittest.main()
