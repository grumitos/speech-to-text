import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types

from config import (
    AVAILABLE_GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODELS,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SEC,
    RETRY_BASE_DELAY,
    TARGET_MIME_TYPE,
)
from models import TranscriptionResult

PLACEHOLDER_API_KEY = "tu_clave_aqui"  # valor de .env.example

TRANSCRIPTION_PROMPT = (
    "Transcribe the following audio exactly as spoken, preserving the original language. "
    "Output only the transcription text, nothing else. No timestamps, no labels, no explanations."
)

# No usamos herramientas; sin esto el SDK avisa por consola del uso directo de AFC.
NO_AUTOMATIC_FUNCTION_CALLING = types.AutomaticFunctionCallingConfig(disable=True)


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

    def max_duration_sec(self) -> Optional[float]:
        """Duración máxima por archivo, o None si el proveedor no la limita."""
        return None


class GeminiProvider(TranscriptionProvider):
    name = "gemini"

    def __init__(self, model: str | None = None):
        self.model_name = model or DEFAULT_GEMINI_MODEL
        if self.model_name not in GEMINI_MODELS:
            available = ", ".join(AVAILABLE_GEMINI_MODELS)
            raise ValueError(f"Modelo '{self.model_name}' no soportado. Disponible: {available}")
        self.spec = GEMINI_MODELS[self.model_name]
        self.client: genai.Client | None = None

    def initialize(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == PLACEHOLDER_API_KEY:
            raise ValueError("GOOGLE_API_KEY no configurada. Añádela al archivo .env")

        # El SDK ni reintenta ni aplica timeout por defecto: sin esto un 429/503 pasajero fallaría
        # el archivo y una conexión colgada bloquearía el lote entero.
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=MAX_RETRIES, initial_delay=RETRY_BASE_DELAY),
            timeout=REQUEST_TIMEOUT_SEC * 1000,  # milisegundos
        )
        self.client = genai.Client(api_key=api_key, http_options=http_options)

        try:
            self.client.models.get(model=f"models/{self.model_name}")
        except Exception as e:
            raise ValueError(f"Error inicializando Gemini: {e}") from e

    def available_models(self) -> List[str]:
        return AVAILABLE_GEMINI_MODELS

    def max_file_size_mb(self) -> float:
        return self.spec.max_file_size_mb

    def max_duration_sec(self) -> Optional[float]:
        return self.spec.max_duration_sec

    def current_model(self) -> str:
        return self.model_name

    def transcribe(
        self,
        audio_path: Path,
        original_filename: str,
        prompt: str = "",
    ) -> TranscriptionResult:
        if self.client is None:
            raise RuntimeError("GeminiProvider no inicializado. Llama a initialize() primero.")
        if prompt and not self.spec.accepts_prompt:
            raise ValueError(f"El modelo {self.model_name} no admite prompt.")

        result = TranscriptionResult(
            file_name=original_filename,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            transcription_text="",
            model_name=self.model_name,
            provider_name=self.name,
        )

        start = time.monotonic()
        try:
            if self.spec.dedicated_transcriber:
                response = self._generate_with_uploaded_audio(audio_path)
            else:
                response = self._generate_with_inline_audio(audio_path, prompt or TRANSCRIPTION_PROMPT)
            result.transcription_text = self._extract_text(response)
        except Exception as e:  # un fallo en un archivo no debe detener el resto del lote
            result.error = f"Error: {e}"
        result.transcription_time = time.monotonic() - start
        return result

    def _generate_with_inline_audio(self, audio_path: Path, prompt: str):
        audio_part = types.Part.from_bytes(data=audio_path.read_bytes(), mime_type=TARGET_MIME_TYPE)
        return self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, audio_part],
            config=types.GenerateContentConfig(
                automatic_function_calling=NO_AUTOMATIC_FUNCTION_CALLING,
            ),
        )

    def _generate_with_uploaded_audio(self, audio_path: Path):
        # Se sube el descriptor y no la ruta: con una ruta el SDK manda el nombre del archivo en una
        # cabecera HTTP que solo admite ASCII, y fallaría con nombres como "reunión.mp3".
        with audio_path.open("rb") as audio_file:
            uploaded = self.client.files.upload(
                file=audio_file,
                config=types.UploadFileConfig(mime_type=TARGET_MIME_TYPE),
            )
        try:
            # Sin language_codes el modelo detecta el idioma; VERBATIM conserva lo dicho tal cual.
            return self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded],
                config=types.GenerateContentConfig(
                    automatic_function_calling=NO_AUTOMATIC_FUNCTION_CALLING,
                    audio_transcription_config=types.AudioTranscriptionConfig(
                        mode=types.AudioTranscriptionConfigMode.VERBATIM,
                    ),
                ),
            )
        finally:
            self._delete_uploaded(uploaded)

    def _delete_uploaded(self, uploaded: types.File) -> None:
        try:
            self.client.files.delete(name=uploaded.name)
        except Exception:
            pass  # el borrado es opcional: la Files API elimina los archivos a las 48 h

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        text = (response.text or "").strip()
        if text:
            return text

        reason = "sin detalle"
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason
            reason = f"finish_reason={getattr(finish_reason, 'name', finish_reason)}"
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            reason = f"block_reason={getattr(block_reason, 'name', block_reason)}"
        raise ValueError(f"Gemini no devolvió texto ({reason}).")


def get_provider(model: str | None = None) -> TranscriptionProvider:
    return GeminiProvider(model=model)
