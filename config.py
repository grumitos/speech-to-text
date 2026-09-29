from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GeminiModel:
    """Capacidades de un modelo Gemini para transcribir audio."""

    id: str
    # Modelo de STT dedicado: se configura con audio_transcription_config y no admite
    # instrucciones de texto (prompt).
    dedicated_transcriber: bool
    max_file_size_mb: float
    # Duración máxima por petición; los audios más largos se dividen en tramos.
    max_duration_sec: Optional[float] = None

    @property
    def accepts_prompt(self) -> bool:
        return not self.dedicated_transcriber


GEMINI_FILES_API_LIMIT_MB = 2048  # límite por archivo de la Files API
# Duración máxima por petición. La documentación admite mucho más (1 h el modelo dedicado, 9,5 h
# los flash), pero con audios de más de ~30 min la calidad cae: el modelo dedicado añade texto
# repetido al final y los flash se saltan partes. Con tramos de ~25 min salen completos.
GEMINI_MAX_DURATION_SEC = 30 * 60

# Configuración de Gemini
GEMINI_MODELS = {
    model.id: model
    for model in (
        GeminiModel(
            id="gemini-3.5-transcribe",
            dedicated_transcriber=True,
            max_file_size_mb=GEMINI_FILES_API_LIMIT_MB,
            max_duration_sec=GEMINI_MAX_DURATION_SEC,
        ),
        GeminiModel(
            id="gemini-3.8-flash",
            dedicated_transcriber=False,
            max_file_size_mb=GEMINI_FILES_API_LIMIT_MB,
            max_duration_sec=GEMINI_MAX_DURATION_SEC,
        ),
        GeminiModel(
            id="gemini-3.5-flash-lite",
            dedicated_transcriber=False,
            max_file_size_mb=GEMINI_FILES_API_LIMIT_MB,
            max_duration_sec=GEMINI_MAX_DURATION_SEC,
        ),
    )
}
DEFAULT_GEMINI_MODEL = "gemini-3.5-transcribe"
AVAILABLE_GEMINI_MODELS = list(GEMINI_MODELS)

# Formatos que Gemini acepta tal cual: extensión -> tipo MIME.
NATIVE_AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
}
# Con FFmpeg solo se envían sin convertir los formatos comprimidos y con el códec esperado;
# el resto (WAV, AIFF, FLAC, vídeos...) se recodifica, lo que además reduce mucho el tamaño.
PASSTHROUGH_CODECS = {
    ".mp3": {"mp3"},
    ".m4a": {"aac"},
    ".aac": {"aac"},
    ".ogg": {"vorbis", "opus"},
    ".opus": {"opus"},
    ".webm": {"vorbis", "opus"},
}

# Audio de destino de la conversión: MP3 mono a 16 kHz, el formato estándar para voz. Gemini
# ya mezcla los canales a mono y reduce la calidad internamente, y así se codifica muy rápido.
TARGET_FORMAT = ".mp3"
TARGET_SAMPLE_RATE = 16000
TARGET_BITRATE = "48k"
SEGMENT_DURATION_MARGIN = 0.9  # los tramos miden el 90 % del máximo por petición
SILENCE_THRESHOLD_DB = -35  # umbral para buscar pausas donde cortar los tramos
SILENCE_MIN_DURATION_SEC = 0.3

MIN_AUDIO_DURATION_SEC = 1.0
FFMPEG_TIMEOUT_SEC = 1800  # una conversión de horas de audio supera con creces los 2 min
FFPROBE_TIMEOUT_SEC = 60
MAX_CONVERSION_WORKERS = 8
REQUEST_TIMEOUT_SEC = 15 * 60  # por petición HTTP; sin esto una conexión colgada bloquea el lote
FILE_PROCESSING_TIMEOUT_SEC = 10 * 60  # espera a que la Files API deje el archivo en ACTIVE
FILE_POLL_INTERVAL_SEC = 2
MAX_RETRIES = 5  # intentos totales ante errores transitorios del servidor
RETRY_BASE_DELAY = 5  # segundos; se duplica en cada intento
RETRYABLE_STATUS_CODES = [408, 500, 502, 503, 504]  # el 429 lo gestiona el proveedor
RATE_LIMIT_RETRIES = 3  # reintentos ante un 429 por límite de tasa por minuto
RATE_LIMIT_MAX_WAIT_SEC = 120  # tope de espera que se acepta de la API

# Directorios por defecto
DEFAULT_INPUT_DIR = Path("entrada")
DEFAULT_OUTPUT_DIR = Path("salida")

# Sin FFmpeg no se puede inspeccionar el contenido: solo se aceptan estas extensiones.
AUDIO_EXTENSIONS = set(NATIVE_AUDIO_MIME_TYPES)
