import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from config import (
    AUDIO_EXTENSIONS,
    FFMPEG_TIMEOUT_SEC,
    FFPROBE_TIMEOUT_SEC,
    MIN_AUDIO_DURATION_SEC,
    NATIVE_AUDIO_MIME_TYPES,
    PASSTHROUGH_CODECS,
    SILENCE_MIN_DURATION_SEC,
    SILENCE_THRESHOLD_DB,
    TARGET_BITRATE,
    TARGET_FORMAT,
    TARGET_SAMPLE_RATE,
)


@dataclass(frozen=True)
class AudioInfo:
    codec: str
    duration_sec: Optional[float]


@dataclass(frozen=True)
class AudioValidation:
    is_valid: bool
    error_message: Optional[str]
    needs_conversion: bool
    duration_sec: Optional[float] = None
    # False cuando el archivo no contiene audio: se ignora sin contarlo como fallo.
    is_audio: bool = True


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def is_candidate_file(path: Path, ffmpeg_available: bool) -> bool:
    """Con FFmpeg cualquier archivo puede ser audio (se decide por su contenido)."""
    if not path.is_file() or path.name.startswith("."):
        return False
    return ffmpeg_available or path.suffix.lower() in AUDIO_EXTENSIONS


def probe_audio(file_path: Path) -> Optional[AudioInfo]:
    """Devuelve el códec y la duración de la primera pista de audio, o None si no hay ninguna."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,duration:format=duration",
        "-of", "json", str(file_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFPROBE_TIMEOUT_SEC,
        )
        data = json.loads(result.stdout or "{}") if result.returncode == 0 else {}
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None

    streams = data.get("streams") or []
    if not streams:
        return None
    duration = _parse_duration(streams[0].get("duration"))
    if duration is None:
        duration = _parse_duration((data.get("format") or {}).get("duration"))
    return AudioInfo(codec=streams[0].get("codec_name") or "", duration_sec=duration)


def _parse_duration(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # ausente o "N/A"
        return None


def validate_audio(
    audio_path: Path,
    max_size_mb: float,
    ffmpeg_available: bool,
    max_duration_sec: Optional[float] = None,
    probe_fn: Callable[[Path], Optional[AudioInfo]] = probe_audio,
) -> AudioValidation:
    extension = audio_path.suffix.lower()
    size_mb = audio_path.stat().st_size / (1024 * 1024)

    if not ffmpeg_available:
        if extension not in NATIVE_AUDIO_MIME_TYPES:
            return AudioValidation(
                is_valid=False,
                error_message=(
                    f"Formato {extension or '(sin extensión)'} requiere conversión, "
                    "pero FFmpeg no está disponible."
                ),
                needs_conversion=False,
            )
        if size_mb > max_size_mb:
            return AudioValidation(
                is_valid=False,
                error_message=f"Archivo demasiado grande ({size_mb:.2f} MB > {max_size_mb:g} MB)",
                needs_conversion=False,
            )
        return AudioValidation(is_valid=True, error_message=None, needs_conversion=False)

    info = probe_fn(audio_path)
    if info is None:
        return AudioValidation(
            is_valid=False,
            error_message="No contiene una pista de audio legible.",
            needs_conversion=False,
            is_audio=False,
        )

    duration = info.duration_sec
    if duration is not None and duration < MIN_AUDIO_DURATION_SEC:
        return AudioValidation(
            is_valid=False,
            error_message=f"Audio demasiado corto ({duration:.2f}s < {MIN_AUDIO_DURATION_SEC}s)",
            needs_conversion=False,
            duration_sec=duration,
        )

    is_passthrough = info.codec in PASSTHROUGH_CODECS.get(extension, set())
    too_big = size_mb > max_size_mb
    too_long = duration is not None and max_duration_sec is not None and duration > max_duration_sec
    return AudioValidation(
        is_valid=True,
        error_message=None,
        needs_conversion=not is_passthrough or too_big or too_long,
        duration_sec=duration,
    )


_SILENCE_START = re.compile(r"silence_start: (-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end: (-?\d+(?:\.\d+)?)")


def _detect_silences(input_path: Path) -> List[Tuple[float, float]]:
    """Intervalos de silencio (inicio, fin) de la primera pista de audio."""
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", str(input_path),
        "-map", "0:a:0",
        "-af", f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={SILENCE_MIN_DURATION_SEC}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, encoding="utf-8", errors="replace", timeout=FFMPEG_TIMEOUT_SEC
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    silences: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for line in result.stderr.splitlines():
        if (match := _SILENCE_START.search(line)) is not None:
            start = float(match.group(1))
        elif (match := _SILENCE_END.search(line)) is not None and start is not None:
            silences.append((start, float(match.group(1))))
            start = None
    return silences


def choose_cut_times(
    silences: List[Tuple[float, float]], duration_sec: float, segment_sec: float
) -> List[float]:
    """Instantes de corte para tramos de como máximo `segment_sec`, en pausas del habla.

    Cada corte es el silencio más tardío que cabe en el tramo (y no antes de su mitad); si no hay
    ninguno se corta en el límite, aunque pueda partir una palabra.
    """
    pauses = [(start + end) / 2 for start, end in silences]
    cuts: List[float] = []
    position = 0.0
    while duration_sec - position > segment_sec:
        limit = position + segment_sec
        candidates = [t for t in pauses if position + segment_sec / 2 <= t <= limit]
        position = max(candidates) if candidates else limit
        cuts.append(position)
    return cuts


def _segment_timing_args(input_path: Path, segment_sec: float) -> List[str]:
    info = probe_audio(input_path)
    if info is not None and info.duration_sec:
        cuts = choose_cut_times(_detect_silences(input_path), info.duration_sec, segment_sec)
        if cuts:
            return ["-segment_times", ",".join(f"{t:.3f}" for t in cuts)]
    return ["-segment_time", str(int(segment_sec))]


def convert_audio(
    input_path: Path,
    output_dir: Path,
    segment_sec: Optional[float] = None,
) -> Tuple[List[Path], Optional[str]]:
    """Convierte a MP3 mono de 16 kHz dentro de `output_dir`.

    Con `segment_sec` divide el resultado en tramos de esa duración. Devuelve los archivos
    generados en orden, o el motivo del fallo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if segment_sec:
        output = output_dir / f"part_%03d{TARGET_FORMAT}"
        segment_args = ["-f", "segment", "-reset_timestamps", "1", *_segment_timing_args(input_path, segment_sec)]
    else:
        output = output_dir / f"part_000{TARGET_FORMAT}"
        segment_args = []

    command = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-map", "0:a:0",  # solo la primera pista de audio: descarta vídeo, subtítulos y datos
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "libmp3lame", "-b:a", TARGET_BITRATE,
        *segment_args,
        str(output),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFMPEG_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        error = f"FFmpeg timeout ({FFMPEG_TIMEOUT_SEC}s)"
    except FileNotFoundError:
        error = "FFmpeg no encontrado."
    except subprocess.CalledProcessError as e:
        error = f"FFmpeg falló: {(e.stderr or '').strip() or e}"
    except Exception as e:
        error = f"Error inesperado: {e}"
    else:
        parts = sorted(output_dir.glob(f"part_*{TARGET_FORMAT}"))
        if parts:
            return parts, None
        error = "FFmpeg finalizó sin generar el archivo convertido."

    shutil.rmtree(output_dir, ignore_errors=True)
    return [], error
