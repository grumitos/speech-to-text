import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio import cleanup_temp_files, convert_audio, validate_audio
from config import TEMP_SUFFIX


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

    def test_validate_audio_rejects_audio_longer_than_provider_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.mp3"
            audio_path.write_bytes(b"a" * 100)

            result = validate_audio(
                audio_path=audio_path,
                max_size_mb=1.0,
                ffmpeg_available=True,
                max_duration_sec=3600,
                duration_fn=lambda _: 3601.0,
            )

            self.assertFalse(result.is_valid)
            self.assertIn("Audio demasiado largo", result.error_message or "")

    def test_validate_audio_accepts_audio_within_provider_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.mp3"
            audio_path.write_bytes(b"a" * 100)

            result = validate_audio(
                audio_path=audio_path,
                max_size_mb=1.0,
                ffmpeg_available=True,
                max_duration_sec=3600,
                duration_fn=lambda _: 3600.0,
            )

            self.assertTrue(result.is_valid)

    def test_convert_audio_writes_to_a_non_audio_temp_file_then_renames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_path = base / "sample.wav"
            source_path.write_bytes(b"source")
            seen = {}

            def fake_run(command, **_kwargs):
                output_path = Path(command[command.index("128k") + 1])
                seen["temp_name"] = output_path.name
                seen["format"] = command[command.index("-f") + 1]
                output_path.write_bytes(b"converted")

            with patch("audio.subprocess.run", side_effect=fake_run):
                converted_path, error = convert_audio(source_path)

            self.assertIsNone(error)
            self.assertEqual(seen["temp_name"], "sample.wav.mp3" + TEMP_SUFFIX)
            self.assertEqual(seen["format"], "mp3")  # sin extensión .mp3 FFmpeg no infiere el formato
            self.assertEqual(sorted(p.name for p in base.iterdir()), ["sample.wav", "sample.wav.mp3"])

    def test_cleanup_temp_files_only_removes_our_partial_conversions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            partial = base / f"clip.wav.mp3{TEMP_SUFFIX}"
            user_files = [base / "notes_temp.mp3", base / "download.mp3.part", base / "voice.mp3"]
            partial.write_bytes(b"partial")
            for path in user_files:
                path.write_bytes(b"user data")

            cleanup_temp_files(base)

            self.assertFalse(partial.exists())
            self.assertTrue(all(path.exists() for path in user_files))

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
