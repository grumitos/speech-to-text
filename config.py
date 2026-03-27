from pathlib import Path

# Configuración de Gemini
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
AVAILABLE_GEMINI_MODELS = [DEFAULT_GEMINI_MODEL]
GEMINI_INLINE_LIMIT_MB = 20
MIN_AUDIO_DURATION_SEC = 1.0
FFMPEG_TIMEOUT_SEC = 120
TARGET_FORMAT = ".mp3"
MAX_CONVERSION_WORKERS = 8
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
