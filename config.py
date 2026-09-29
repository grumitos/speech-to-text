from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GeminiModel:
    """Capacidades de un modelo Gemini para transcribir audio."""

    id: str
    # Modelo de STT dedicado: se configura con audio_transcription_config, recibe el
    # audio por Files API y no admite instrucciones de texto (prompt).
    dedicated_transcriber: bool
    max_file_size_mb: float
    max_duration_sec: Optional[float] = None

    @property
    def accepts_prompt(self) -> bool:
        return not self.dedicated_transcriber


GEMINI_INLINE_LIMIT_MB = 20  # límite de la petición con audio en línea
GEMINI_FILES_API_LIMIT_MB = 2048  # límite por archivo de la Files API
GEMINI_TRANSCRIBE_MAX_DURATION_SEC = 60 * 60  # 1 h por petición

# Configuración de Gemini
GEMINI_MODELS = {
    model.id: model
    for model in (
        GeminiModel(
            id="gemini-3.5-transcribe",
            dedicated_transcriber=True,
            max_file_size_mb=GEMINI_FILES_API_LIMIT_MB,
            max_duration_sec=GEMINI_TRANSCRIBE_MAX_DURATION_SEC,
        ),
        GeminiModel(
            id="gemini-3.8-flash",
            dedicated_transcriber=False,
            max_file_size_mb=GEMINI_INLINE_LIMIT_MB,
        ),
        GeminiModel(
            id="gemini-3.5-flash-lite",
            dedicated_transcriber=False,
            max_file_size_mb=GEMINI_INLINE_LIMIT_MB,
        ),
    )
}
DEFAULT_GEMINI_MODEL = "gemini-3.5-transcribe"
AVAILABLE_GEMINI_MODELS = list(GEMINI_MODELS)

MIN_AUDIO_DURATION_SEC = 1.0
FFMPEG_TIMEOUT_SEC = 600  # una conversión de horas de audio supera con creces los 2 min
TARGET_FORMAT = ".mp3"
TARGET_MIME_TYPE = "audio/mpeg"
TEMP_SUFFIX = ".stt-tmp"  # archivos parciales de la conversión con FFmpeg
MAX_CONVERSION_WORKERS = 8
REQUEST_TIMEOUT_SEC = 15 * 60  # por petición HTTP; sin esto una conexión colgada bloquea el lote
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # segundos

# Directorios por defecto
DEFAULT_INPUT_DIR = Path("entrada")
DEFAULT_OUTPUT_DIR = Path("salida")

# Extensiones de audio aceptadas
AUDIO_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".flac",
    ".ogg", ".aac", ".wma", ".opus", ".webm",
    ".aiff", ".mpeg", ".mpga",
}
