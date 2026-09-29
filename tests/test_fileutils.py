import tempfile
import unittest
from pathlib import Path

from fileutils import unique_path


class UniquePathTests(unittest.TestCase):
    def test_returns_the_same_path_when_it_is_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav.mp3"

            self.assertEqual(unique_path(path), path)

    def test_adds_a_counter_before_the_extension_until_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "clip.wav.mp3").write_bytes(b"1")
            (base / "clip.wav.1.mp3").write_bytes(b"2")

            self.assertEqual(unique_path(base / "clip.wav.mp3"), base / "clip.wav.2.mp3")


if __name__ == "__main__":
    unittest.main()
