import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import errors, types

from config import (
    AVAILABLE_GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL,
    FILE_POLL_INTERVAL_SEC,
    FILE_PROCESSING_TIMEOUT_SEC,
    GEMINI_MODELS,
    MAX_RETRIES,
    NATIVE_AUDIO_MIME_TYPES,
    RATE_LIMIT_MAX_WAIT_SEC,
    RATE_LIMIT_RETRIES,
    REQUEST_TIMEOUT_SEC,
    RETRY_BASE_DELAY,
    RETRYABLE_STATUS_CODES,
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

        # El SDK ni reintenta ni aplica timeout por defecto: sin esto un 503 pasajero fallaría
        # el archivo y una conexión colgada bloquearía el lote entero.
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=MAX_RETRIES,
                initial_delay=RETRY_BASE_DELAY,
                http_status_codes=RETRYABLE_STATUS_CODES,
            ),
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
            response = self._generate(audio_path, prompt or TRANSCRIPTION_PROMPT)
            result.transcription_text = self._extract_text(response)
        except Exception as e:  # un fallo en un archivo no debe detener el resto del lote
            result.error = f"Error: {e}"
        result.transcription_time = time.monotonic() - start
        return result

    def _generate(self, audio_path: Path, prompt: str) -> types.GenerateContentResponse:
        mime_type = self._mime_type_for(audio_path)
        # Se sube el descriptor y no la ruta: con una ruta el SDK manda el nombre del archivo en una
        # cabecera HTTP que solo admite ASCII, y fallaría con nombres como "reunión.mp3".
        with audio_path.open("rb") as audio_file:
            uploaded = self.client.files.upload(
                file=audio_file,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
        try:
            uploaded = self._wait_until_active(uploaded)
            if self.spec.dedicated_transcriber:
                # Sin language_codes el modelo detecta el idioma; VERBATIM conserva lo dicho tal cual.
                contents = [uploaded]
                config = types.GenerateContentConfig(
                    automatic_function_calling=NO_AUTOMATIC_FUNCTION_CALLING,
                    audio_transcription_config=types.AudioTranscriptionConfig(
                        mode=types.AudioTranscriptionConfigMode.VERBATIM,
                    ),
                )
            else:
                contents = [prompt, uploaded]
                config = types.GenerateContentConfig(
                    automatic_function_calling=NO_AUTOMATIC_FUNCTION_CALLING,
                )
            return self._generate_content(model=self.model_name, contents=contents, config=config)
        finally:
            self._delete_uploaded(uploaded)

    def _generate_content(self, **request) -> types.GenerateContentResponse:
        """Reintenta un 429 por límite por minuto esperando lo que indica la API.

        Los tramos de un audio largo se envían seguidos y pueden agotar ese límite: el backoff
        corto del SDK no alcanza, hay que esperar a que se renueve la ventana.
        """
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return self.client.models.generate_content(**request)
            except errors.APIError as e:
                wait = _rate_limit_wait_sec(e)
                if wait is None or wait > RATE_LIMIT_MAX_WAIT_SEC or attempt == RATE_LIMIT_RETRIES:
                    raise
                time.sleep(wait + 1)
        raise AssertionError("inalcanzable")  # el bucle siempre retorna o relanza

    @staticmethod
    def _mime_type_for(audio_path: Path) -> str:
        extension = audio_path.suffix.lower()
        if extension not in NATIVE_AUDIO_MIME_TYPES:
            raise ValueError(f"Formato de audio no soportado por Gemini: {extension or '(sin extensión)'}")
        return NATIVE_AUDIO_MIME_TYPES[extension]

    def _wait_until_active(self, uploaded: types.File) -> types.File:
        """Los archivos grandes pueden quedar un rato en PROCESSING antes de poder usarse."""
        deadline = time.monotonic() + FILE_PROCESSING_TIMEOUT_SEC
        while uploaded.state == types.FileState.PROCESSING:
            if time.monotonic() > deadline:
                raise TimeoutError("El archivo subido no terminó de procesarse a tiempo.")
            time.sleep(FILE_POLL_INTERVAL_SEC)
            uploaded = self.client.files.get(name=uploaded.name)
        if uploaded.state == types.FileState.FAILED:
            raise ValueError("Gemini no pudo procesar el archivo subido.")
        return uploaded

    def _delete_uploaded(self, uploaded: types.File) -> None:
        try:
            self.client.files.delete(name=uploaded.name)
        except Exception:
            pass  # el borrado es opcional: la Files API elimina los archivos a las 48 h

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        """Junta el texto de la respuesta.

        El modelo de transcripción dedicado no devuelve partes de texto sino de tipo
        `audio_transcription`, así que `response.text` sale vacío con él.
        """
        candidate = response.candidates[0] if response.candidates else None
        parts = (candidate.content.parts if candidate and candidate.content else None) or []
        chunks = []
        for part in parts:
            if part.audio_transcription and part.audio_transcription.text:
                chunks.append(part.audio_transcription.text)
            elif part.text and not part.thought:
                chunks.append(part.text)

        finish_reason = candidate.finish_reason if candidate else None
        if finish_reason == types.FinishReason.MAX_TOKENS:
            raise ValueError("La transcripción se truncó por el límite de salida del modelo (MAX_TOKENS).")

        text = "".join(chunks).strip()
        if text:
            return text

        reason = "sin detalle"
        if finish_reason:
            reason = f"finish_reason={getattr(finish_reason, 'name', finish_reason)}"
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            reason = f"block_reason={getattr(block_reason, 'name', block_reason)}"
        raise ValueError(f"Gemini no devolvió texto ({reason}).")


def _rate_limit_wait_sec(error: errors.APIError) -> Optional[float]:
    """Segundos que pide esperar la API ante un 429 por límite por minuto, o None si no aplica.

    Un límite diario o un 429 sin indicación de espera no se arreglan esperando unos segundos.
    """
    if error.code != 429 or not isinstance(error.details, dict):
        return None
    details = [d for d in (error.details.get("error") or {}).get("details") or [] if isinstance(d, dict)]
    per_minute = any(
        "PerMinute" in (violation.get("quotaId") or "")
        for detail in details
        for violation in detail.get("violations") or []
    )
    if not per_minute:
        return None
    for detail in details:
        try:
            return float(str(detail["retryDelay"]).removesuffix("s"))
        except (KeyError, ValueError):
            continue
    return None


def get_provider(model: str | None = None) -> TranscriptionProvider:
    return GeminiProvider(model=model)
