from pathlib import Path

# Límites de archivo por proveedor
OPENAI_FILE_LIMIT_MB = 25
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

# Modelos disponibles por proveedor
PROVIDER_MODELS = {
    "gemini": {
        "default": "gemini-3-flash-preview",
        "models": [
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
        ],
    },
    "openai": {
        "default": "gpt-4o-transcribe",
        "models": [
            "gpt-4o-transcribe",
            "gpt-4o-transcribe-diarize",
            "gpt-4o-mini-transcribe",
        ],
    },
}

# Extensiones de audio aceptadas
AUDIO_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".flac",
    ".ogg", ".aac", ".wma", ".opus", ".webm",
    ".aiff", ".mpeg", ".mpga",
}
