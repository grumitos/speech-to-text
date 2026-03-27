import tempfile
import unittest
from pathlib import Path

from models import TranscriptionResult
from ui import write_transcription_file


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

            write_transcription_file(wav_result, output_dir)
            write_transcription_file(mp3_result, output_dir)

            self.assertTrue((output_dir / "msg" / "clip.wav.txt").exists())
            self.assertTrue((output_dir / "msg" / "clip.mp3.txt").exists())


if __name__ == "__main__":
    unittest.main()
