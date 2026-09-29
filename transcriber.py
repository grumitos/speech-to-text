import os
import shutil
import tempfile
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from audio import (
    AudioValidation,
    check_ffmpeg,
    convert_audio,
    is_candidate_file,
    validate_audio,
)
from config import MAX_CONVERSION_WORKERS, SEGMENT_DURATION_MARGIN
from fileutils import unique_path
from models import TranscriptionResult
from providers import TranscriptionProvider
from ui import (
    make_progress,
    print_bullet,
    print_kv,
    print_section,
    print_state,
    write_transcription_file,
    STYLE_DEFAULT,
    STYLE_SUCCESS,
    STYLE_ERROR,
    STYLE_INFO,
    STYLE_WARNING,
)


@dataclass(frozen=True)
class PreparedFile:
    original_path: Path
    # Lo que se envía a Gemini: el propio original o los tramos resultantes de convertirlo.
    audio_paths: Tuple[Path, ...]
    error_message: Optional[str] = None


@dataclass(frozen=True)
class PendingConversion:
    path: Path
    segment_sec: Optional[float]


ValidationFn = Callable[[Path, float, bool, Optional[float]], AudioValidation]
ConversionFn = Callable[[Path, Path, Optional[float]], Tuple[List[Path], Optional[str]]]
ResultWriterFn = Callable[[TranscriptionResult, Path], None]
MoveFileFn = Callable[[str, str], str]
NowFn = Callable[[], datetime]


class Transcriber:
    def __init__(
        self,
        provider: TranscriptionProvider,
        input_dir: Path,
        output_dir: Path,
        ffmpeg_checker: Callable[[], bool] = check_ffmpeg,
        audio_validator: ValidationFn = validate_audio,
        audio_converter: ConversionFn = convert_audio,
        result_writer: ResultWriterFn = write_transcription_file,
        file_mover: MoveFileFn = shutil.move,
        now_fn: NowFn = datetime.now,
    ):
        self.provider = provider
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.audio_validator = audio_validator
        self.audio_converter = audio_converter
        self.result_writer = result_writer
        self.file_mover = file_mover
        self.now_fn = now_fn
        self.ffmpeg_available = ffmpeg_checker()

        if self.ffmpeg_available:
            print_state("success", "FFmpeg disponible")
        else:
            print_state(
                "disabled",
                "FFmpeg no encontrado: solo se admiten formatos que Gemini acepta directamente, "
                "sin conversión ni división de audios largos.",
            )

    def process_files(self, prompt: str = "") -> int:
        """Procesa el directorio de entrada y devuelve la cantidad de archivos con error."""
        self._print_start(prompt)

        all_files = self._collect_files()
        if not all_files:
            print_state("empty", "No se encontraron archivos de audio en el directorio de entrada.")
            return 0

        # Las conversiones se hacen en un directorio temporal: los originales no se tocan y solo
        # se archivan cuando su transcripción sale bien.
        with tempfile.TemporaryDirectory(prefix="speech-to-text-") as work_dir:
            prepared_files, pending, skipped_count = self._validate_files(all_files)
            prepared_files.extend(self._convert_files(pending, Path(work_dir)))

            if not prepared_files:
                print_state("empty", "No hay archivos válidos para transcribir.")
                return skipped_count

            conversion_errors = [f for f in prepared_files if f.error_message]
            for file in conversion_errors:
                self._write_conversion_error(file)

            valid_files = [f for f in prepared_files if not f.error_message]
            success_count, transcription_error_count = self._transcribe_files(
                valid_files=valid_files,
                prompt=prompt,
            )

        error_count = len(conversion_errors) + transcription_error_count
        self._print_summary(success_count=success_count, error_count=error_count)
        return error_count + skipped_count

    def _print_start(self, prompt: str) -> None:
        print_section("Procesamiento")
        print_kv("Proveedor", self.provider.name, STYLE_INFO)
        if prompt:
            print_kv("Prompt", prompt, STYLE_DEFAULT)
        print_kv("Entrada", self.input_dir)
        print_kv("Salida", self.output_dir)

    def _collect_files(self) -> List[Path]:
        candidates = (f for f in self.input_dir.iterdir() if is_candidate_file(f, self.ffmpeg_available))
        return sorted(candidates, key=lambda p: p.name.lower())

    def _validate_files(
        self, all_files: List[Path]
    ) -> Tuple[List[PreparedFile], List[PendingConversion], int]:
        print_section(f"Fase 1 / Validación ({len(all_files)} archivos)")
        prepared_files: List[PreparedFile] = []
        pending: List[PendingConversion] = []
        invalid_messages: List[str] = []
        ignored_names: List[str] = []
        max_size = self.provider.max_file_size_mb()
        max_duration = self.provider.max_duration_sec()

        with make_progress("Validando...") as progress:
            task = progress.add_task("", total=len(all_files))
            for file_path in all_files:
                validation = self.audio_validator(file_path, max_size, self.ffmpeg_available, max_duration)

                if not validation.is_audio:
                    ignored_names.append(file_path.name)
                elif not validation.is_valid:
                    invalid_messages.append(f"{file_path.name}: {validation.error_message}")
                elif validation.needs_conversion:
                    segment_sec = self._segment_duration(validation.duration_sec, max_duration)
                    pending.append(PendingConversion(file_path, segment_sec))
                else:
                    prepared_files.append(PreparedFile(file_path, (file_path,)))
                progress.advance(task)

        if ignored_names:
            print_state("warning", "Ignorados (no contienen audio)")
            for name in ignored_names:
                print_bullet(name, STYLE_WARNING)

        if invalid_messages:
            print_state("error", "Archivos omitidos")
            for msg in invalid_messages:
                print_bullet(msg, STYLE_ERROR)

        return prepared_files, pending, len(invalid_messages)

    @staticmethod
    def _segment_duration(duration: Optional[float], max_duration: Optional[float]) -> Optional[float]:
        """Duración de cada tramo si el audio supera el máximo por petición del modelo."""
        if duration is None or max_duration is None or duration <= max_duration:
            return None
        return max_duration * SEGMENT_DURATION_MARGIN

    def _convert_files(self, pending: List[PendingConversion], work_dir: Path) -> List[PreparedFile]:
        if not pending:
            print_section("Fase 2 / Conversión")
            print_state("disabled", "No se requiere conversión.")
            return []

        print_section(f"Fase 2 / Conversión ({len(pending)} archivos a MP3 mono 16 kHz)")

        converted_files: List[PreparedFile] = []
        converted_count = 0
        split_count = 0
        failed_count = 0

        with make_progress("Convirtiendo...") as progress:
            task = progress.add_task("", total=len(pending))
            workers = min(os.cpu_count() or 4, MAX_CONVERSION_WORKERS)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.audio_converter, item.path, work_dir / f"{index:04d}", item.segment_sec
                    ): item
                    for index, item in enumerate(pending)
                }
                for future in concurrent.futures.as_completed(futures):
                    original = futures[future].path
                    parts, conv_error = future.result()

                    if parts:
                        converted_files.append(PreparedFile(original, tuple(parts)))
                        converted_count += 1
                        split_count += len(parts) > 1
                    else:
                        converted_files.append(
                            PreparedFile(original, (), error_message=conv_error or "Error de conversión.")
                        )
                        failed_count += 1

                    progress.advance(task)

        if converted_count:
            print_state("success", f"Convertidos: {converted_count} archivos")
        if split_count:
            print_state("info", f"Divididos en tramos por su duración: {split_count} archivos")
        if failed_count:
            print_state("error", f"Fallos: {failed_count} archivos")

        return converted_files

    def _write_conversion_error(self, file: PreparedFile) -> None:
        result = TranscriptionResult(
            file_name=file.original_path.name,
            date=self.now_fn().strftime("%Y-%m-%d %H:%M:%S"),
            transcription_text="",
            model_name=self.provider.current_model(),
            provider_name=self.provider.name,
            conversion_error=file.error_message or "Error de conversión.",
        )
        self.result_writer(result, self.output_dir)

    def _transcribe_files(
        self,
        valid_files: List[PreparedFile],
        prompt: str,
    ) -> Tuple[int, int]:
        print_section(f"Fase 3 / Transcripción ({len(valid_files)} archivos)")
        print_kv("Proveedor", self.provider.name, STYLE_INFO)

        success_count = 0
        error_count = 0

        with make_progress(f"Transcribiendo con {self.provider.name}...") as progress:
            task = progress.add_task("", total=len(valid_files))

            for file in valid_files:
                result = self._transcribe_parts(file, prompt)

                if result.error is None:
                    move_error = self._move_processed_audio(file.original_path)
                    if move_error is None:
                        success_count += 1
                    else:
                        error_count += 1
                        result.postprocess_warning = (
                            f"No se pudo mover el audio procesado: {move_error}"
                        )
                else:
                    error_count += 1

                self.result_writer(result, self.output_dir)
                progress.advance(task)

        return success_count, error_count

    def _transcribe_parts(self, file: PreparedFile, prompt: str) -> TranscriptionResult:
        """Transcribe cada tramo en orden y devuelve un único resultado; el primer error lo aborta."""
        total = len(file.audio_paths)
        results: List[TranscriptionResult] = []
        for index, audio_path in enumerate(file.audio_paths, start=1):
            result = self.provider.transcribe(
                audio_path=audio_path,
                original_filename=file.original_path.name,
                prompt=prompt,
            )
            if result.error is not None:
                if total > 1:
                    result.error = f"Tramo {index}/{total}: {result.error}"
                return result
            results.append(result)

        merged = results[0]
        if total > 1:
            merged.transcription_text = "\n".join(r.transcription_text for r in results)
            merged.transcription_time = sum(r.transcription_time for r in results)
        return merged

    def _move_processed_audio(self, origin: Path) -> Optional[str]:
        try:
            archive_dir = self.output_dir / "audio"
            archive_dir.mkdir(parents=True, exist_ok=True)
            # shutil.move sobrescribe en silencio: no debe pisar un audio ya archivado.
            destination = unique_path(archive_dir / origin.name)
            self.file_mover(str(origin), str(destination))
            return None
        except Exception as e:
            print_state("error", f"Error al mover {origin.name}: {e}")
            return str(e)

    def _print_summary(self, success_count: int, error_count: int) -> None:
        print_section("Resumen")
        if success_count:
            print_state("success", f"Transcripciones exitosas: {success_count} archivos")
        if error_count:
            print_state("error", f"Transcripciones fallidas: {error_count} archivos")
        print_kv("Salida de textos", self.output_dir / "msg", STYLE_SUCCESS)
