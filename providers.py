import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

from config import (
    AVAILABLE_GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL,
    GEMINI_INLINE_LIMIT_MB,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)
from models import TranscriptionResult

MIME_TYPES = {
    ".mp3": "audio/mp3",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".aiff": "audio/aiff",
    ".wma": "audio/x-ms-wma",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
}

TRANSCRIPTION_PROMPT = (
    "Transcribe the following audio exactly as spoken, preserving the original language. "
    "Output only the transcription text, nothing else. No timestamps, no labels, no explanations."
)


class TranscriptionProvider(ABC):
    name: str

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        original_filename: str,
        prompt: str = "",
    ) -> TranscriptionResult: ...

    @abstractmethod
    def available_models(self) -> List[str]: ...

    @abstractmethod
    def max_file_size_mb(self) -> float: ...

    @abstractmethod
    def current_model(self) -> str: ...


class GeminiProvider(TranscriptionProvider):
    name = "gemini"

    def __init__(self, model: str | None = None):
        self.model_name = model or DEFAULT_GEMINI_MODEL
        if self.model_name not in AVAILABLE_GEMINI_MODELS:
            available = ", ".join(AVAILABLE_GEMINI_MODELS)
            raise ValueError(f"Modelo '{self.model_name}' no soportado. Disponible: {available}")
        self.client = None

    def initialize(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no configurada. Añádela al archivo .env")

        from google import genai

        self.client = genai.Client(api_key=api_key)

        try:
            self.client.models.get(model=f"models/{self.model_name}")
        except Exception as e:
            raise ValueError(f"Error inicializando Gemini: {e}")

    def available_models(self) -> List[str]:
        return AVAILABLE_GEMINI_MODELS

    def max_file_size_mb(self) -> float:
        return GEMINI_INLINE_LIMIT_MB

    def current_model(self) -> str:
        return self.model_name

    def transcribe(
        self,
        audio_path: Path,
        original_filename: str,
        prompt: str = "",
    ) -> TranscriptionResult:
        from google.genai import types
        if self.client is None:
            raise RuntimeError("GeminiProvider no inicializado. Llama a initialize() primero.")

        result = TranscriptionResult(
            file_name=original_filename,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            transcription_text="",
            model_name=self.model_name,
            provider_name=self.name,
        )

        ext = audio_path.suffix.lower()
        mime_type = MIME_TYPES.get(ext, "audio/mpeg")
        user_prompt = prompt if prompt else TRANSCRIPTION_PROMPT

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.time()

                audio_bytes = audio_path.read_bytes()
                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[user_prompt, audio_part],
                )

                result.transcription_time = time.time() - start
                result.transcription_text = response.text.strip()
                return result

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate" in error_str or "quota" in error_str

                if is_rate_limit and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    time.sleep(delay)
                else:
                    result.error = f"Error: {e}"
                    return result

        return result


def get_provider(model: str | None = None) -> TranscriptionProvider:
    return GeminiProvider(model=model)
