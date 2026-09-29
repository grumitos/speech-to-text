import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from google import genai
from google.genai import types

from config import (
    AVAILABLE_GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL,
    GEMINI_INLINE_LIMIT_MB,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SEC,
    RETRY_BASE_DELAY,
    TARGET_MIME_TYPE,
)
from providers import TRANSCRIPTION_PROMPT, GeminiProvider, get_provider

DEDICATED_MODEL = "gemini-3.5-transcribe"
PROMPT_MODEL = "gemini-3.8-flash"


def make_response(text="hola mundo", finish_reason=None, block_reason=None):
    candidates = [SimpleNamespace(finish_reason=finish_reason)] if finish_reason else []
    feedback = SimpleNamespace(block_reason=block_reason) if block_reason else None
    return SimpleNamespace(text=text, candidates=candidates, prompt_feedback=feedback)


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

    def test_limits_follow_the_transport_of_each_model(self) -> None:
        dedicated = GeminiProvider(DEDICATED_MODEL)
        general = GeminiProvider(PROMPT_MODEL)

        self.assertGreater(dedicated.max_file_size_mb(), GEMINI_INLINE_LIMIT_MB)
        self.assertEqual(dedicated.max_duration_sec(), 3600)
        self.assertEqual(general.max_file_size_mb(), GEMINI_INLINE_LIMIT_MB)
        self.assertIsNone(general.max_duration_sec())


class InitializeTests(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            GeminiProvider().initialize()

    def test_example_placeholder_is_not_a_valid_api_key(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "tu_clave_aqui"}),                 patch("providers.genai.Client") as client_cls,                 self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY no configurada"):
            GeminiProvider().initialize()

        client_cls.assert_not_called()

    def test_client_retries_transient_errors_has_a_timeout_and_checks_the_model(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
                patch("providers.genai.Client") as client_cls:
            provider = GeminiProvider(PROMPT_MODEL)
            provider.initialize()

        retry = client_cls.call_args.kwargs["http_options"].retry_options
        self.assertEqual(retry.attempts, MAX_RETRIES)
        self.assertEqual(retry.initial_delay, RETRY_BASE_DELAY)
        self.assertEqual(client_cls.call_args.kwargs["http_options"].timeout, REQUEST_TIMEOUT_SEC * 1000)
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

    def _provider(self, model: str, response=None, error: Exception | None = None) -> GeminiProvider:
        provider = GeminiProvider(model)
        provider.client = MagicMock()
        self.uploaded = SimpleNamespace(name="files/abc123")

        def fake_upload(**kwargs):
            self.uploaded_bytes = kwargs["file"].read()  # el archivo se cierra al salir del upload
            return self.uploaded

        provider.client.files.upload.side_effect = fake_upload
        generate = provider.client.models.generate_content
        if error is not None:
            generate.side_effect = error
        else:
            generate.return_value = response or make_response()
        return provider

    def test_requires_initialization(self) -> None:
        with self.assertRaises(RuntimeError):
            GeminiProvider().transcribe(self.audio_path, "voice.mp3")

    def test_dedicated_model_uploads_audio_and_uses_transcription_config(self) -> None:
        provider = self._provider(DEDICATED_MODEL, make_response("  texto dictado \n"))

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIsNone(result.error)
        self.assertEqual(result.transcription_text, "texto dictado")
        self.assertEqual(result.model_name, DEDICATED_MODEL)

        upload = provider.client.files.upload.call_args.kwargs
        self.assertEqual(self.uploaded_bytes, b"mp3-bytes")  # se sube el descriptor, no la ruta
        self.assertEqual(upload["config"].mime_type, TARGET_MIME_TYPE)

        request = provider.client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], DEDICATED_MODEL)
        self.assertEqual(request["contents"], [self.uploaded])
        transcription = request["config"].audio_transcription_config
        self.assertEqual(transcription.mode, types.AudioTranscriptionConfigMode.VERBATIM)
        self.assertIsNone(transcription.language_codes)  # detección automática de idioma
        self.assertTrue(request["config"].automatic_function_calling.disable)
        provider.client.files.delete.assert_called_once_with(name="files/abc123")

    def test_dedicated_model_deletes_uploaded_file_when_generation_fails(self) -> None:
        provider = self._provider(DEDICATED_MODEL, error=RuntimeError("503 UNAVAILABLE"))

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("503 UNAVAILABLE", result.error or "")
        provider.client.files.delete.assert_called_once_with(name="files/abc123")

    def test_failed_cleanup_of_uploaded_file_does_not_fail_the_transcription(self) -> None:
        provider = self._provider(DEDICATED_MODEL)
        provider.client.files.delete.side_effect = RuntimeError("no se pudo borrar")

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIsNone(result.error)
        self.assertEqual(result.transcription_text, "hola mundo")

    def test_dedicated_model_rejects_prompts(self) -> None:
        provider = self._provider(DEDICATED_MODEL)

        with self.assertRaises(ValueError):
            provider.transcribe(self.audio_path, "voice.mp3", prompt="con puntuación")

        provider.client.models.generate_content.assert_not_called()

    def test_general_model_sends_inline_audio_with_default_prompt(self) -> None:
        provider = self._provider(PROMPT_MODEL)

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIsNone(result.error)
        provider.client.files.upload.assert_not_called()
        request = provider.client.models.generate_content.call_args.kwargs
        prompt, audio = request["contents"]
        self.assertEqual(request["model"], PROMPT_MODEL)
        self.assertEqual(prompt, TRANSCRIPTION_PROMPT)
        self.assertEqual(audio.inline_data.data, b"mp3-bytes")
        self.assertEqual(audio.inline_data.mime_type, TARGET_MIME_TYPE)
        self.assertTrue(request["config"].automatic_function_calling.disable)

    def test_general_model_uses_custom_prompt(self) -> None:
        provider = self._provider("gemini-3.5-flash-lite")

        provider.transcribe(self.audio_path, "voice.mp3", prompt="Termina cada frase con punto")

        prompt, _audio = provider.client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(prompt, "Termina cada frase con punto")

    def test_empty_response_is_reported_with_its_reason(self) -> None:
        provider = self._provider(
            PROMPT_MODEL,
            make_response(text=None, finish_reason=SimpleNamespace(name="SAFETY")),
        )

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("finish_reason=SAFETY", result.error or "")
        self.assertEqual(result.transcription_text, "")

    def test_blocked_prompt_is_reported(self) -> None:
        provider = self._provider(
            PROMPT_MODEL,
            make_response(text="", block_reason=SimpleNamespace(name="PROHIBITED_CONTENT")),
        )

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertIn("block_reason=PROHIBITED_CONTENT", result.error or "")

    def test_api_errors_are_recorded_per_file(self) -> None:
        provider = self._provider(PROMPT_MODEL, error=RuntimeError("429 RESOURCE_EXHAUSTED"))

        result = provider.transcribe(self.audio_path, "voice.mp3")

        self.assertEqual(result.error, "Error: 429 RESOURCE_EXHAUSTED")
        self.assertGreaterEqual(result.transcription_time, 0)


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
            file_info = {"name": "files/abc", "uri": "https://example.test/files/abc", "mimeType": "audio/mpeg"}
            return httpx.Response(200, headers={"x-goog-upload-status": "final"}, json={"file": file_info})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        reply = {"candidates": [{"content": {"parts": [{"text": "hola"}], "role": "model"}, "finishReason": "STOP"}]}
        return httpx.Response(200, json=reply)

    def _provider(self, model: str) -> GeminiProvider:
        provider = GeminiProvider(model)
        transport = httpx.MockTransport(self._handler)
        provider.client = genai.Client(
            api_key="test-key",
            http_options=types.HttpOptions(httpx_client=httpx.Client(transport=transport)),
        )
        return provider

    def test_dedicated_model_uploads_files_whose_names_are_not_ascii(self) -> None:
        for name in ("reunión.m4a.mp3", "日本語 ñandú.mp3"):
            with self.subTest(name=name):
                self.requests.clear()
                audio_path = Path(self._tmp.name) / name
                audio_path.write_bytes(b"ID3audio")

                result = self._provider(DEDICATED_MODEL).transcribe(audio_path, name)

                self.assertIsNone(result.error)
                self.assertEqual(result.transcription_text, "hola")
                start = next(r for r in self.requests if "start" in r.headers.get("x-goog-upload-command", ""))
                self.assertNotIn("x-goog-upload-file-name", start.headers)  # el nombre no sale del equipo
                self.assertEqual(json.loads(start.content)["file"]["size_bytes"], len(b"ID3audio"))
                self.assertEqual(self.requests[-1].method, "DELETE")

    def test_transcription_request_has_the_expected_json_body(self) -> None:
        audio_path = Path(self._tmp.name) / "voice.mp3"
        audio_path.write_bytes(b"ID3audio")

        self._provider(DEDICATED_MODEL).transcribe(audio_path, "voice.mp3")

        generate = next(r for r in self.requests if r.url.path.endswith(":generateContent"))
        body = json.loads(generate.content)
        self.assertEqual(body["generationConfig"]["audioTranscriptionConfig"], {"mode": "VERBATIM"})
        self.assertEqual(body["contents"][0]["parts"][0]["fileData"]["mime_type"], TARGET_MIME_TYPE)


if __name__ == "__main__":
    unittest.main()
