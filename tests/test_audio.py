import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speech_to_text.audio import (
    AudioInfo,
    _detect_silences,
    check_ffmpeg,
    choose_cut_times,
    convert_audio,
    is_candidate_file,
    probe_audio,
    validate_audio,
)


def make_file(directory: str, name: str, size: int = 100) -> Path:
    path = Path(directory) / name
    path.write_bytes(b"a" * size)
    return path


class CandidateFileTests(unittest.TestCase):
    def test_with_ffmpeg_any_visible_file_is_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(is_candidate_file(make_file(tmp, "clip.mkv"), ffmpeg_available=True))
            self.assertTrue(is_candidate_file(make_file(tmp, "sin_extension"), ffmpeg_available=True))
            self.assertFalse(is_candidate_file(make_file(tmp, ".gitkeep"), ffmpeg_available=True))
            self.assertFalse(is_candidate_file(Path(tmp), ffmpeg_available=True))

    def test_without_ffmpeg_only_formats_gemini_accepts_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(is_candidate_file(make_file(tmp, "voice.M4A"), ffmpeg_available=False))
            self.assertTrue(is_candidate_file(make_file(tmp, "voice.wav"), ffmpeg_available=False))
            self.assertFalse(is_candidate_file(make_file(tmp, "voice.wma"), ffmpeg_available=False))


class ValidateWithoutFfmpegTests(unittest.TestCase):
    def test_accepts_formats_gemini_takes_as_is(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.mp3", "a.wav", "a.flac", "a.m4a", "a.ogg"):
                with self.subTest(name=name):
                    result = validate_audio(make_file(tmp, name), max_size_mb=1.0, ffmpeg_available=False)

                    self.assertTrue(result.is_valid)
                    self.assertFalse(result.needs_conversion)

    def test_rejects_other_formats_because_they_need_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_audio(make_file(tmp, "voice.wma"), max_size_mb=1.0, ffmpeg_available=False)

            self.assertFalse(result.is_valid)
            self.assertIn("FFmpeg", result.error_message or "")

    def test_rejects_files_over_the_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = make_file(tmp, "voice.mp3", size=2 * 1024 * 1024)

            result = validate_audio(path, max_size_mb=1.0, ffmpeg_available=False)

            self.assertFalse(result.is_valid)
            self.assertIn("demasiado grande", result.error_message or "")


class ValidateWithFfmpegTests(unittest.TestCase):
    def _validate(self, name: str, codec: str = "mp3", duration=60.0, size: int = 100, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_file(tmp, name, size)
            return validate_audio(
                path,
                max_size_mb=kwargs.pop("max_size_mb", 1.0),
                ffmpeg_available=True,
                probe_fn=lambda _: AudioInfo(codec=codec, duration_sec=duration),
                **kwargs,
            )

    def test_compressed_formats_with_the_expected_codec_are_sent_as_is(self) -> None:
        for name, codec in (("a.mp3", "mp3"), ("a.m4a", "aac"), ("a.aac", "aac"),
                            ("a.ogg", "vorbis"), ("a.opus", "opus"), ("a.webm", "opus")):
            with self.subTest(name=name):
                result = self._validate(name, codec)

                self.assertTrue(result.is_valid)
                self.assertFalse(result.needs_conversion)

    def test_everything_else_is_converted(self) -> None:
        cases = (
            ("a.wav", "pcm_s16le"),  # sin comprimir: convertirlo lo reduce mucho
            ("a.flac", "flac"),
            ("a.aiff", "pcm_s16be"),
            ("a.wma", "wmav2"),
            ("a.mp4", "aac"),  # vídeo: solo interesa su pista de audio
            ("a.mkv", "opus"),
            ("a.m4a", "alac"),  # extensión válida pero códec no admitido
            ("a.mp3", "aac"),  # extensión que no corresponde al contenido
            ("sin_extension", "mp3"),
        )
        for name, codec in cases:
            with self.subTest(name=name, codec=codec):
                self.assertTrue(self._validate(name, codec).needs_conversion)

    def test_files_without_an_audio_stream_are_flagged_as_not_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_audio(
                make_file(tmp, "notas.txt"), 1.0, True, probe_fn=lambda _: None
            )

            self.assertFalse(result.is_valid)
            self.assertFalse(result.is_audio)

    def test_rejects_too_short_audio(self) -> None:
        result = self._validate("a.mp3", duration=0.2)

        self.assertFalse(result.is_valid)
        self.assertTrue(result.is_audio)
        self.assertIn("Audio demasiado corto", result.error_message or "")

    def test_audio_longer_than_the_model_limit_is_converted_to_be_split(self) -> None:
        result = self._validate("a.mp3", duration=3601.0, max_duration_sec=3600)

        self.assertTrue(result.is_valid)
        self.assertTrue(result.needs_conversion)
        self.assertEqual(result.duration_sec, 3601.0)

    def test_audio_within_the_model_limit_is_left_alone(self) -> None:
        result = self._validate("a.mp3", duration=3600.0, max_duration_sec=3600)

        self.assertFalse(result.needs_conversion)

    def test_oversized_file_is_converted_instead_of_rejected(self) -> None:
        result = self._validate("a.mp3", size=2 * 1024 * 1024, max_size_mb=1.0)

        self.assertTrue(result.is_valid)
        self.assertTrue(result.needs_conversion)

    def test_unknown_duration_does_not_block_the_file(self) -> None:
        result = self._validate("a.mp3", duration=None, max_duration_sec=3600)

        self.assertTrue(result.is_valid)
        self.assertFalse(result.needs_conversion)


class ProbeAudioTests(unittest.TestCase):
    def _probe(self, stdout: str = "", returncode: int = 0, error: Exception | None = None):
        completed = SimpleNamespace(stdout=stdout, returncode=returncode)
        with patch("speech_to_text.audio.subprocess.run", side_effect=error, return_value=completed):
            return probe_audio(Path("x.mp3"))

    def test_reads_codec_and_duration_of_the_first_audio_stream(self) -> None:
        info = self._probe('{"streams": [{"codec_name": "aac", "duration": "12.5"}], "format": {"duration": "13"}}')

        self.assertEqual(info, AudioInfo(codec="aac", duration_sec=12.5))

    def test_falls_back_to_the_container_duration(self) -> None:
        info = self._probe('{"streams": [{"codec_name": "opus", "duration": "N/A"}], "format": {"duration": "7.25"}}')

        self.assertEqual(info, AudioInfo(codec="opus", duration_sec=7.25))

    def test_duration_may_be_unknown(self) -> None:
        info = self._probe('{"streams": [{"codec_name": "mp3"}], "format": {}}')

        self.assertEqual(info, AudioInfo(codec="mp3", duration_sec=None))

    def test_returns_none_without_audio_stream_or_on_failure(self) -> None:
        self.assertIsNone(self._probe('{"streams": [], "format": {"duration": "3"}}'))
        self.assertIsNone(self._probe("", returncode=1))
        self.assertIsNone(self._probe("no es json"))
        self.assertIsNone(self._probe(error=subprocess.TimeoutExpired("ffprobe", 1)))
        self.assertIsNone(self._probe(error=FileNotFoundError()))


class CutPointTests(unittest.TestCase):
    def test_no_cuts_when_the_audio_fits_in_one_segment(self) -> None:
        self.assertEqual(choose_cut_times([(5.0, 6.0)], duration_sec=100, segment_sec=100), [])

    def test_cuts_at_the_latest_pause_inside_the_segment(self) -> None:
        silences = [(30.0, 31.0), (80.0, 82.0), (95.0, 97.0), (120.0, 121.0)]

        cuts = choose_cut_times(silences, duration_sec=150, segment_sec=100)

        self.assertEqual(cuts, [96.0])  # el silencio de 95-97 s, no el de 120 s (fuera del tramo)

    def test_falls_back_to_the_limit_when_there_are_no_pauses(self) -> None:
        self.assertEqual(choose_cut_times([], duration_sec=250, segment_sec=100), [100.0, 200.0])

    def test_ignores_pauses_in_the_first_half_of_the_segment(self) -> None:
        cuts = choose_cut_times([(10.0, 12.0)], duration_sec=150, segment_sec=100)

        self.assertEqual(cuts, [100.0])

    def test_every_segment_stays_within_the_limit(self) -> None:
        silences = [(i * 37.0, i * 37.0 + 1.0) for i in range(1, 30)]

        cuts = choose_cut_times(silences, duration_sec=1000, segment_sec=200)

        edges = [0.0, *cuts, 1000.0]
        self.assertTrue(all(b - a <= 200 for a, b in zip(edges, edges[1:])))

    def test_detects_silences_from_ffmpeg_output(self) -> None:
        stderr = (
            "[silencedetect @ 0x1] silence_start: 1.5\n"
            "[silencedetect @ 0x1] silence_end: 2.25 | silence_duration: 0.75\n"
            "[silencedetect @ 0x1] silence_start: 9\n"
        )
        with patch("speech_to_text.audio.subprocess.run", return_value=SimpleNamespace(stderr=stderr)):
            self.assertEqual(_detect_silences(Path("x.mp3")), [(1.5, 2.25)])  # el último no termina

    def test_detecting_silences_never_raises(self) -> None:
        with patch("speech_to_text.audio.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 1)):
            self.assertEqual(_detect_silences(Path("x.mp3")), [])


class ConvertAudioTests(unittest.TestCase):
    def _convert(self, output_names, segment_sec=None, probe=None, silences=()):
        """Ejecuta convert_audio con FFmpeg simulado; devuelve (resultado, comando, directorio)."""
        seen = {}
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"

            def fake_run(command, **_kwargs):
                seen["command"] = command
                for name in output_names:
                    (out_dir / name).write_bytes(b"mp3")
                return SimpleNamespace()

            with patch("speech_to_text.audio.subprocess.run", side_effect=fake_run), \
                    patch("speech_to_text.audio.probe_audio", return_value=probe), \
                    patch("speech_to_text.audio._detect_silences", return_value=list(silences)):
                result = convert_audio(Path(tmp) / "in.wav", out_dir, segment_sec)
            return result, seen["command"], out_dir

    def test_single_file_is_a_small_mono_16khz_mp3(self) -> None:
        (parts, error), command, _ = self._convert(["part_000.mp3"])

        self.assertIsNone(error)
        self.assertEqual([p.name for p in parts], ["part_000.mp3"])
        self.assertIn("-nostdin", command)  # en lote, FFmpeg no debe leer la entrada estándar
        self.assertEqual(command[command.index("-map") + 1], "0:a:0")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-b:a") + 1], "48k")
        self.assertNotIn("segment", command)

    def test_segments_are_cut_at_the_chosen_pauses(self) -> None:
        probe = AudioInfo(codec="pcm_s16le", duration_sec=250.0)
        (parts, error), command, _ = self._convert(
            ["part_000.mp3", "part_001.mp3", "part_002.mp3"],
            segment_sec=100,
            probe=probe,
            silences=[(90.0, 92.0), (190.0, 192.0)],
        )

        self.assertIsNone(error)
        self.assertEqual(len(parts), 3)
        self.assertEqual(command[command.index("-segment_times") + 1], "91.000,191.000")
        self.assertIn("part_%03d.mp3", command[-1])

    def test_segments_fall_back_to_fixed_length_when_the_duration_is_unknown(self) -> None:
        (_, error), command, _ = self._convert(["part_000.mp3"], segment_sec=100.7, probe=None)

        self.assertIsNone(error)
        self.assertEqual(command[command.index("-segment_time") + 1], "100")
        self.assertNotIn("-segment_times", command)

    def _failing(self, error: Exception):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with patch("speech_to_text.audio.subprocess.run", side_effect=error):
                result = convert_audio(Path(tmp) / "in.wav", out_dir)
            return result, out_dir.exists()

    def test_ffmpeg_errors_are_reported_and_leave_no_partial_output(self) -> None:
        failure = subprocess.CalledProcessError(1, "ffmpeg", stderr="Invalid data found")

        (parts, error), still_exists = self._failing(failure)

        self.assertEqual(parts, [])
        self.assertIn("Invalid data found", error or "")
        self.assertFalse(still_exists)

    def test_timeout_and_missing_ffmpeg_are_reported(self) -> None:
        (_, timeout_error), _ = self._failing(subprocess.TimeoutExpired("ffmpeg", 1))
        (_, missing_error), _ = self._failing(FileNotFoundError())

        self.assertIn("timeout", timeout_error or "")
        self.assertIn("no encontrado", missing_error or "")

    def test_success_without_output_is_an_error(self) -> None:
        (parts, error), _, _ = self._convert([])

        self.assertEqual(parts, [])
        self.assertIn("sin generar", error or "")


@unittest.skipUnless(check_ffmpeg(), "requiere FFmpeg y ffprobe")
class RealFfmpegTests(unittest.TestCase):
    """Ejercita FFmpeg de verdad con audio sintético generado al vuelo."""

    def _tone(self, directory: Path, name: str, seconds: int, args=()) -> Path:
        path = directory / name
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"sine=frequency=440:duration={seconds}", *args, str(path)],
            check=True,
        )
        return path

    def test_probe_validate_and_convert_a_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._tone(Path(tmp), "reunión.wav", 3, ["-ac", "2", "-ar", "44100"])

            validation = validate_audio(source, max_size_mb=100, ffmpeg_available=True)
            parts, error = convert_audio(source, Path(tmp) / "out")

            self.assertTrue(validation.needs_conversion)  # PCM: se recodifica
            self.assertAlmostEqual(validation.duration_sec, 3.0, delta=0.1)
            self.assertIsNone(error)
            converted = probe_audio(parts[0])
            self.assertEqual(converted.codec, "mp3")
            self.assertAlmostEqual(converted.duration_sec, 3.0, delta=0.2)
            self.assertLess(parts[0].stat().st_size, source.stat().st_size / 10)
            self.assertTrue(source.exists())  # el original no se toca

    def test_a_video_yields_its_audio_and_a_text_file_is_not_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            video = base / "clip.mp4"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=2",
                 "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)],
                check=True,
            )
            notes = make_file(tmp, "notas.txt")

            parts, error = convert_audio(video, base / "out")

            self.assertIsNone(error)
            self.assertEqual(probe_audio(parts[0]).codec, "mp3")
            self.assertFalse(validate_audio(notes, 1.0, True).is_audio)

    def test_a_long_recording_is_split_into_segments_within_the_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._tone(Path(tmp), "long.wav", 9)

            parts, error = convert_audio(source, Path(tmp) / "out", segment_sec=4)

            self.assertIsNone(error)
            self.assertEqual(len(parts), 3)
            self.assertLessEqual(probe_audio(parts[0]).duration_sec, 4.3)

    def test_conversion_of_a_file_without_audio_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            notes = make_file(tmp, "notas.txt")

            parts, error = convert_audio(notes, Path(tmp) / "out")

            self.assertEqual(parts, [])
            self.assertIn("FFmpeg falló", error or "")


if __name__ == "__main__":
    unittest.main()
