import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from google import genai
from google.genai import errors, types

from config import (
    AVAILABLE_GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL,
    GEMINI_FILES_API_LIMIT_MB,
    MAX_RETRIES,
    RATE_LIMIT_MAX_WAIT_SEC,
    RATE_LIMIT_RETRIES,
    REQUEST_TIMEOUT_SEC,
    RETRY_BASE_DELAY,
    RETRYABLE_STATUS_CODES,
)
from providers import TRANSCRIPTION_PROMPT, GeminiProvider, get_provider

DEDICATED_MODEL = "gemini-3.5-transcribe"
PROMPT_MODEL = "gemini-3.8-flash"


def _response(parts=(), finish_reason=types.FinishReason.STOP, block_reason=None):
    candidates = []
    if finish_reason is not None or parts:
        content = types.Content(role="model", parts=list(parts)) if parts else None
        candidates = [types.Candidate(content=content, finish_reason=finish_reason)]
    feedback = types.GenerateContentResponsePromptFeedback(block_reason=block_reason) if block_reason else None
    return types.GenerateContentResponse(candidates=candidates, prompt_feedback=feedback)


def transcription_response(*texts: str, **kwargs):
    """Respuesta como la del modelo dedicado: partes `audio_transcription`, sin partes de texto."""
    return _response([types.Part(audio_transcription=types.Transcription(text=t)) for t in texts], **kwargs)


def text_response(*texts: str, **kwargs):
    return _response([types.Part(text=t) for t in texts], **kwargs)


class ModelCatalogTests(unittest.TestCase):
    def test_default_is_the_dedicated_transcriber(self) -> None:
        provider = GeminiProvider()

        self.assertEqual(DEFAULT_GEMINI_MODEL, DEDICATED_MODEL)
        self.assertEqual(provider.current_model(), DEDICATED_MODEL)

    def test_current_models_are_available(self) -> None:
        self.assertEqual(
            AVAILABLE_GEMINI_MODELS,
            [DEDICATED_MODEL, PROMPT_MODEL, "gemini-3.5-flash-lite"],
        )
        self.assertEqual(GeminiProvider().available_models(), AVAILABLE_GEMINI_MODELS)

    def test_rejects_retired_and_unknown_models(self) -> None:
        for model in ("gemini-3.1-flash-lite-preview", "gemini-2.5-flash", "gpt-4o"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                get_provider(model=model)

    def test_every_model_takes_files_up_to_the_files_api_limit(self) -> None:
        for model in AVAILABLE_GEMINI_MODELS:
            with self.subTest(model=model):
                self.assertEqual(GeminiProvider(model).max_file_size_mb(), GEMINI_FILES_API_LIMIT_MB)

    def test_every_model_transcribes_at_most_30_minutes_per_request(self) -> None:
        for model in AVAILABLE_GEMINI_MODELS:
            with self.subTest(model=model):
                self.assertEqual(GeminiProvider(model).max_duration_sec(), 30 * 60)


class InitializeTests(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            GeminiProvider().initialize()

    def test_example_placeholder_is_not_a_valid_api_key(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "tu_clave_aqui"}), \
                patch("providers.genai.Client") as client_cls, \
                self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY no configurada"):
            GeminiProvider().initialize()

        client_cls.assert_not_called()

    def test_client_retries_transient_errors_has_a_timeout_and_checks_the_model(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
                patch("providers.genai.Client") as client_cls:
            provider = GeminiProvider(PROMPT_MODEL)
            provider.initialize()

        http_options = client_cls.call_args.kwargs["http_options"]
        self.assertEqual(http_options.retry_options.attempts, MAX_RETRIES)
        self.assertEqual(http_options.retry_options.initial_delay, RETRY_BASE_DELAY)
        self.assertEqual(http_options.retry_options.http_status_codes, RETRYABLE_STATUS_CODES)
        self.assertNotIn(429, RETRYABLE_STATUS_CODES)  # los límites de tasa se esperan aparte
        self.assertEqual(http_options.timeout, REQUEST_TIMEOUT_SEC * 1000)
        client_cls.return_value.models.get.assert_called_once_with(model=f"models/{PROMPT_MODEL}")

    def test_wraps_model_lookup_failures(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
                patch("providers.genai.Client") as client_cls:
            client_cls.return_value.models.get.side_effect = RuntimeError("404 NOT_FOUND")

            with self.assertRaisesRegex(ValueError, "404 NOT_FOUND"):
                GeminiProvider().initialize()


class TranscribeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.audio_path = Path(self._tmp.name) / "voice.mp3"
        self.audio_path.write_bytes(b"mp3-bytes")
        self.uploaded = SimpleNamespace(name="files/abc123", state=None)
        self.upload_calls: list[dict] = []

    def _provider(self, model: str, response=None, error: Exception | None = None) -> GeminiProvider:
        provider = GeminiProvider(model)
        provider.client = MagicMock()

        def fake_upload(**kwargs):
            # El archivo se cierra al terminar la subida: hay que leerlo aquí.
            self.upload_calls.append({"bytes": kwargs["file"].read(), "mime": kwargs["config"].mime_type})
            return self.uploaded

        provider.client.files.upload.side_effect = fake_upload
        generate = provider.client.models.generate_content
        if error is not None:
            generate.side_effect = error
        else:
            generate.return_value = response or text_response("hola mundo")
        return provider

    def test_requires_initialization(self) -> None:
        with self.assertRaises(RuntimeError):
            GeminiProvider().transcribe(self.audio_path, "voice.mp3")

    def test_dedicated_model_uploads_audio_and_uses_transcription_config(self) -> None:
        provider = self._provider(DEDICATED_MODEL, transcription_response("  texto dictado \n"))

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIsNone(result.error)
        self.assertEqual(result.transcription_text, "texto dictado")
        self.assertEqual(result.model_name, DEDICATED_MODEL)
        self.assertEqual(self.upload_calls, [{"bytes": b"mp3-bytes", "mime": "audio/mpeg"}])

        request = provider.client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], DEDICATED_MODEL)
        self.assertEqual(request["contents"], [self.uploaded])
        transcription = request["config"].audio_transcription_config
        self.assertEqual(transcription.mode, types.AudioTranscriptionConfigMode.VERBATIM)
        self.assertIsNone(transcription.language_codes)  # detección automática de idioma
        self.assertTrue(request["config"].automatic_function_calling.disable)
        provider.client.files.delete.assert_called_once_with(name="files/abc123")

    def test_dedicated_model_text_is_not_in_response_text(self) -> None:
        response = transcription_response("solo en audio_transcription")

        with redirect_stderr(io.StringIO()):  # el SDK avisa por log de que hay partes que no son texto
            self.assertFalse(response.text)  # el atajo del SDK sale vacío con este modelo...

        provider = self._provider(DEDICATED_MODEL, response)
        self.assertEqual(provider.transcribe(self.audio_path, "voice.mp3").transcription_text,
                         "solo en audio_transcription")  # ...y aun así se lee la transcripción

    def test_general_models_also_upload_and_send_the_prompt_with_the_file(self) -> None:
        provider = self._provider(PROMPT_MODEL, text_response("hola mundo"))

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIsNone(result.error)
        self.assertEqual(result.transcription_text, "hola mundo")
        self.assertEqual(self.upload_calls, [{"bytes": b"mp3-bytes", "mime": "audio/mpeg"}])
        request = provider.client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], PROMPT_MODEL)
        self.assertEqual(request["contents"], [TRANSCRIPTION_PROMPT, self.uploaded])
        self.assertTrue(request["config"].automatic_function_calling.disable)
        self.assertIsNone(request["config"].audio_transcription_config)
        provider.client.files.delete.assert_called_once_with(name="files/abc123")

    def test_general_model_uses_custom_prompt(self) -> None:
        provider = self._provider("gemini-3.5-flash-lite")

        provider.transcribe(self.audio_path, "voice.mp3", prompt="Termina cada frase con punto")

        contents = provider.client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(contents, ["Termina cada frase con punto", self.uploaded])

    def test_general_model_ignores_thought_parts(self) -> None:
        response = _response([types.Part(text="razonando...", thought=True), types.Part(text="La transcripción")])
        provider = self._provider(PROMPT_MODEL, response)

        self.assertEqual(provider.transcribe(self.audio_path, "voice.mp3").transcription_text, "La transcripción")

    def test_dedicated_model_rejects_prompts(self) -> None:
        provider = self._provider(DEDICATED_MODEL)

        with self.assertRaises(ValueError):
            provider.transcribe(self.audio_path, "voice.mp3", prompt="con puntuación")

        provider.client.models.generate_content.assert_not_called()

    def test_mime_type_follows_the_file_extension(self) -> None:
        expected = {"a.m4a": "audio/mp4", "a.wav": "audio/wav", "a.OGG": "audio/ogg", "a.flac": "audio/flac",
                    "a.webm": "audio/webm", "a.aac": "audio/aac", "a.aiff": "audio/aiff"}
        provider = self._provider(PROMPT_MODEL)
        for name, mime in expected.items():
            with self.subTest(name=name):
                path = Path(self._tmp.name) / name
                path.write_bytes(b"x")
                self.upload_calls.clear()

                provider.transcribe(path, name)

                self.assertEqual(self.upload_calls[0]["mime"], mime)

    def test_unsupported_extension_is_reported_without_uploading(self) -> None:
        path = Path(self._tmp.name) / "voice.wma"
        path.write_bytes(b"x")
        provider = self._provider(PROMPT_MODEL)

        result = provider.transcribe(path, "voice.wma")

        self.assertIn("no soportado", result.error or "")
        provider.client.files.upload.assert_not_called()

    def test_waits_for_large_files_to_leave_the_processing_state(self) -> None:
        processing = SimpleNamespace(name="files/abc123", state=types.FileState.PROCESSING)
        active = SimpleNamespace(name="files/abc123", state=types.FileState.ACTIVE)
        provider = self._provider(PROMPT_MODEL)
        self.uploaded = processing
        provider.client.files.get.side_effect = [processing, active]

        with patch("providers.time.sleep") as sleep:
            result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIsNone(result.error)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(provider.client.models.generate_content.call_args.kwargs["contents"][1], active)

    def test_processing_failures_and_timeouts_are_reported_and_cleaned_up(self) -> None:
        failed = SimpleNamespace(name="files/abc123", state=types.FileState.FAILED)
        provider = self._provider(PROMPT_MODEL)
        self.uploaded = failed

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("no pudo procesar", result.error or "")
        provider.client.models.generate_content.assert_not_called()
        provider.client.files.delete.assert_called_once_with(name="files/abc123")

        stuck = SimpleNamespace(name="files/abc123", state=types.FileState.PROCESSING)
        self.uploaded = stuck
        with patch("providers.FILE_PROCESSING_TIMEOUT_SEC", -1):
            result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("no terminó de procesarse", result.error or "")

    def test_uploaded_file_is_deleted_when_generation_fails(self) -> None:
        provider = self._provider(DEDICATED_MODEL, error=RuntimeError("503 UNAVAILABLE"))

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("503 UNAVAILABLE", result.error or "")
        provider.client.files.delete.assert_called_once_with(name="files/abc123")

    def test_failed_cleanup_of_uploaded_file_does_not_fail_the_transcription(self) -> None:
        provider = self._provider(DEDICATED_MODEL, transcription_response("hola"))
        provider.client.files.delete.side_effect = RuntimeError("no se pudo borrar")

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIsNone(result.error)
        self.assertEqual(result.transcription_text, "hola")

    def test_empty_response_is_reported_with_its_reason(self) -> None:
        provider = self._provider(PROMPT_MODEL, _response(finish_reason=types.FinishReason.SAFETY))

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("finish_reason=SAFETY", result.error or "")
        self.assertEqual(result.transcription_text, "")

    def test_blocked_prompt_is_reported(self) -> None:
        response = _response(finish_reason=None, block_reason=types.BlockedReason.PROHIBITED_CONTENT)
        provider = self._provider(PROMPT_MODEL, response)

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("block_reason=PROHIBITED_CONTENT", result.error or "")

    def test_truncated_output_is_an_error_not_a_silent_partial_transcript(self) -> None:
        response = text_response("mitad de la transcripción", finish_reason=types.FinishReason.MAX_TOKENS)
        provider = self._provider(PROMPT_MODEL, response)

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("MAX_TOKENS", result.error or "")

    def test_api_errors_are_recorded_per_file(self) -> None:
        provider = self._provider(PROMPT_MODEL, error=RuntimeError("429 RESOURCE_EXHAUSTED"))

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertEqual(result.error, "Error: 429 RESOURCE_EXHAUSTED")
        self.assertGreaterEqual(result.transcription_time, 0)


def rate_limit_error(quota_id: str = "GenerateContentInputTokensPerModelPerMinute", delay: str | None = "12.5s"):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}]
    if delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "límite", "details": details}}
    return errors.ClientError(429, body)


class RateLimitRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.audio_path = Path(self._tmp.name) / "voice.mp3"
        self.audio_path.write_bytes(b"mp3-bytes")

    def _transcribe(self, side_effect):
        provider = GeminiProvider(DEDICATED_MODEL)
        provider.client = MagicMock()
        provider.client.files.upload.return_value = SimpleNamespace(name="files/abc", state=None)
        provider.client.models.generate_content.side_effect = side_effect
        with patch("providers.time.sleep") as sleep:
            result = provider.transcribe(self.audio_path, "voice.mp3")
        return result, provider.client.models.generate_content.call_count, sleep

    def test_per_minute_limit_waits_for_the_time_the_api_asks_and_retries(self) -> None:
        result, calls, sleep = self._transcribe([rate_limit_error(), transcription_response("hola")])

        self.assertIsNone(result.error)
        self.assertEqual(result.transcription_text, "hola")
        self.assertEqual(calls, 2)
        sleep.assert_called_once_with(13.5)  # 12,5 s pedidos + 1 s de margen

    def test_gives_up_after_the_configured_number_of_retries(self) -> None:
        result, calls, sleep = self._transcribe([rate_limit_error()] * (RATE_LIMIT_RETRIES + 1))

        self.assertIn("429", result.error or "")
        self.assertEqual(calls, RATE_LIMIT_RETRIES + 1)
        self.assertEqual(sleep.call_count, RATE_LIMIT_RETRIES)

    def test_limits_that_waiting_a_few_seconds_cannot_fix_are_not_retried(self) -> None:
        cases = {
            "daily": rate_limit_error(quota_id="GenerateRequestsPerDayPerProjectPerModel"),
            "no wait hint": rate_limit_error(delay=None),
            "wait too long": rate_limit_error(delay=f"{RATE_LIMIT_MAX_WAIT_SEC + 1}s"),
            "not a rate limit": errors.ServerError(503, {"error": {"code": 503, "message": "caído"}}),
        }
        for label, error in cases.items():
            with self.subTest(label=label):
                result, calls, sleep = self._transcribe([error])

                self.assertIsNotNone(result.error)
                self.assertEqual(calls, 1)
                sleep.assert_not_called()


class SdkWireTests(unittest.TestCase):
    """Usa el SDK real con un transporte HTTP simulado: valida lo que los mocks no ven."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.requests: list[httpx.Request] = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        command = request.headers.get("x-goog-upload-command", "")
        if "start" in command:
            return httpx.Response(
                200,
                headers={
                    "x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/files/s1",
                    "x-goog-upload-status": "active",
                },
                json={},
            )
        if "finalize" in command:
            file_info = {"name": "files/abc", "uri": "https://example.test/files/abc",
                         "mimeType": "audio/mpeg", "state": "ACTIVE"}
            return httpx.Response(200, headers={"x-goog-upload-status": "final"}, json={"file": file_info})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        # Así responde el modelo dedicado: el texto va en `audioTranscription`, no en `text`.
        part = {"audioTranscription": {"text": "hola"}} if "transcribe" in request.url.path else {"text": "hola"}
        reply = {"candidates": [{"content": {"parts": [part], "role": "model"}, "finishReason": "STOP"}]}
        return httpx.Response(200, json=reply)

    def _provider(self, model: str) -> GeminiProvider:
        provider = GeminiProvider(model)
        transport = httpx.MockTransport(self._handler)
        provider.client = genai.Client(
            api_key="test-key",
            http_options=types.HttpOptions(httpx_client=httpx.Client(transport=transport)),
        )
        return provider

    def _audio(self, name: str) -> Path:
        path = Path(self._tmp.name) / name
        path.write_bytes(b"ID3audio")
        return path

    def test_both_kinds_of_model_read_the_transcript_and_clean_up(self) -> None:
        for model in (DEDICATED_MODEL, PROMPT_MODEL):
            with self.subTest(model=model):
                self.requests.clear()

                result = self._provider(model).transcribe(self._audio("voice.mp3"), "voice.mp3")

                self.assertIsNone(result.error)
                self.assertEqual(result.transcription_text, "hola")
                self.assertEqual(self.requests[-1].method, "DELETE")

    def test_uploads_files_whose_names_are_not_ascii(self) -> None:
        for model in (DEDICATED_MODEL, PROMPT_MODEL):
            for name in ("reunión.m4a.mp3", "日本語 ñandú.mp3"):
                with self.subTest(model=model, name=name):
                    self.requests.clear()

                    result = self._provider(model).transcribe(self._audio(name), name)

                    self.assertIsNone(result.error)
                    start = next(r for r in self.requests if "start" in r.headers.get("x-goog-upload-command", ""))
                    self.assertNotIn("x-goog-upload-file-name", start.headers)  # el nombre no sale del equipo
                    self.assertEqual(json.loads(start.content)["file"]["size_bytes"], len(b"ID3audio"))

    def test_dedicated_request_has_the_expected_json_body(self) -> None:
        self._provider(DEDICATED_MODEL).transcribe(self._audio("voice.mp3"), "voice.mp3")

        generate = next(r for r in self.requests if r.url.path.endswith(":generateContent"))
        body = json.loads(generate.content)
        self.assertEqual(body["generationConfig"]["audioTranscriptionConfig"], {"mode": "VERBATIM"})
        self.assertEqual(body["contents"][0]["parts"][0]["fileData"]["mime_type"], "audio/mpeg")

    def test_general_request_sends_the_prompt_and_a_file_reference_not_inline_audio(self) -> None:
        self._provider(PROMPT_MODEL).transcribe(self._audio("voice.mp3"), "voice.mp3", prompt="Con puntuación")

        generate = next(r for r in self.requests if r.url.path.endswith(":generateContent"))
        parts = json.loads(generate.content)["contents"][0]["parts"]
        self.assertEqual(parts[0], {"text": "Con puntuación"})
        self.assertIn("fileData", parts[1])
        self.assertNotIn("inlineData", parts[1])


if __name__ == "__main__":
    unittest.main()
