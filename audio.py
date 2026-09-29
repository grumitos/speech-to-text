import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from config import (
    MIN_AUDIO_DURATION_SEC,
    FFMPEG_TIMEOUT_SEC,
    TARGET_FORMAT,
    TEMP_SUFFIX,
    AUDIO_EXTENSIONS,
)
from fileutils import unique_path


@dataclass(frozen=True)
class AudioValidation:
    is_valid: bool
    error_message: Optional[str]
    needs_conversion: bool
    warning_message: Optional[str] = None


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def get_duration(file_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(file_path), "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
        if match:
            h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
            return h * 3600 + m * 60 + s
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def validate_audio(
    audio_path: Path,
    max_size_mb: float,
    ffmpeg_available: bool,
    max_duration_sec: Optional[float] = None,
    duration_fn: Callable[[Path], Optional[float]] = get_duration,
) -> AudioValidation:
    needs_conversion = audio_path.suffix.lower() != TARGET_FORMAT

    if needs_conversion and not ffmpeg_available:
        return AudioValidation(
            is_valid=False,
            error_message=f"Formato {audio_path.suffix} requiere conversión, pero FFmpeg no está disponible.",
            needs_conversion=False,
        )

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb * 1.1:
        return AudioValidation(
            is_valid=False,
            error_message=f"Archivo demasiado grande ({file_size_mb:.2f} MB > ~{max_size_mb} MB)",
            needs_conversion=needs_conversion,
        )

    warning_message = None
    if file_size_mb > max_size_mb * 0.9:
        warning_message = (
            f"{audio_path.name} ({file_size_mb:.2f} MB) está cerca del límite de {max_size_mb} MB."
        )

    if ffmpeg_available:
        duration = duration_fn(audio_path)
        if duration is not None and duration < MIN_AUDIO_DURATION_SEC:
            return AudioValidation(
                is_valid=False,
                error_message=f"Audio demasiado corto ({duration:.2f}s < {MIN_AUDIO_DURATION_SEC}s)",
                needs_conversion=needs_conversion,
            )
        if duration is not None and max_duration_sec is not None and duration > max_duration_sec:
            return AudioValidation(
                is_valid=False,
                error_message=(
                    f"Audio demasiado largo ({duration / 60:.1f} min > {max_duration_sec / 60:.0f} min)"
                ),
                needs_conversion=needs_conversion,
            )

    return AudioValidation(
        is_valid=True,
        error_message=None,
        needs_conversion=needs_conversion,
        warning_message=warning_message,
    )


def convert_audio(input_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    final_output = _converted_output_path(input_path)
    temp_output = final_output.with_name(final_output.name + TEMP_SUFFIX)

    command = [
        "ffmpeg", "-y", "-i", str(input_path), "-vn",
        "-f", TARGET_FORMAT.lstrip("."),
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(temp_output), "-hide_banner", "-loglevel", "error",
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
        if temp_output.exists():
            temp_output.rename(final_output)
        if not final_output.exists():
            return None, "FFmpeg finalizó sin generar el archivo convertido."
        return final_output, None

    except subprocess.TimeoutExpired:
        _cleanup(temp_output)
        return None, f"FFmpeg timeout ({FFMPEG_TIMEOUT_SEC}s)"

    except FileNotFoundError:
        return None, "FFmpeg no encontrado."

    except subprocess.CalledProcessError as e:
        _cleanup(temp_output)
        stderr = (e.stderr or "").strip()
        return None, f"FFmpeg falló: {stderr or e}"

    except Exception as e:
        _cleanup(temp_output)
        return None, f"Error inesperado: {e}"


def cleanup_temp_files(directory: Path) -> None:
    """Elimina los archivos parciales que dejó una conversión interrumpida."""
    for path in directory.glob(f"*{TARGET_FORMAT}{TEMP_SUFFIX}"):
        _cleanup(path)


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _converted_output_path(input_path: Path) -> Path:
    source_suffix = input_path.suffix.lower().lstrip(".") or "audio"
    return unique_path(input_path.with_name(f"{input_path.stem}.{source_suffix}{TARGET_FORMAT}"))
