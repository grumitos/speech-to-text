import os
import shutil
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from audio import AudioValidation, check_ffmpeg, validate_audio, convert_audio, is_audio_file
from config import MAX_CONVERSION_WORKERS, TARGET_FORMAT
from models import TranscriptionResult
from providers import TranscriptionProvider
from ui import (
    console,
    make_progress,
    write_transcription_file,
    style_keyword,
    style_success,
    style_error,
    style_info,
    STYLE_DEFAULT,
    STYLE_SUCCESS,
    STYLE_ERROR,
    STYLE_INFO,
)


@dataclass(frozen=True)
class PreparedFile:
    original_path: Path
    transcribe_path: Optional[Path]
    error_message: Optional[str] = None


ValidationFn = Callable[[Path, float, bool], AudioValidation]
ConversionFn = Callable[[Path], Tuple[Optional[Path], Optional[str]]]
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
            console.print(f"| FFmpeg disponible", style=STYLE_INFO)
        else:
            console.print(
                f"| {style_error('Advertencia:')} FFmpeg no encontrado. La conversión de formatos no funcionará.",
                style=STYLE_ERROR,
            )

    def process_files(self, prompt: str = "") -> None:
        self._print_start(prompt)

        all_files = self._collect_audio_files()
        if not all_files:
            console.print("| No se encontraron archivos de audio en el directorio de entrada.", style=STYLE_INFO)
            return

        self._cleanup_temp_files()
        prepared_files, files_needing_conversion = self._validate_files(all_files)
        prepared_files.extend(self._convert_files(files_needing_conversion))

        if not prepared_files:
            console.print("\n| No hay archivos válidos para transcribir.", style=STYLE_INFO)
            return

        conversion_errors = [f for f in prepared_files if f.error_message]
        for file in conversion_errors:
            self._write_conversion_error(file)

        valid_files = [f for f in prepared_files if f.transcribe_path is not None]
        success_count, transcription_error_count = self._transcribe_files(
            valid_files=valid_files,
            prompt=prompt,
        )

        error_count = len(conversion_errors) + transcription_error_count
        self._print_summary(success_count=success_count, error_count=error_count)

    def _print_start(self, prompt: str) -> None:
        console.print(f"\n| Iniciando procesamiento con {style_keyword(self.provider.name)}", style=STYLE_INFO)
        if prompt:
            console.print(f"| Prompt: {style_keyword(prompt)}", style=STYLE_INFO)
        console.print(f"| Entrada: {self.input_dir}", style=STYLE_DEFAULT)
        console.print(f"| Salida:  {self.output_dir}", style=STYLE_DEFAULT)

    def _collect_audio_files(self) -> List[Path]:
        return sorted((f for f in self.input_dir.iterdir() if is_audio_file(f)), key=lambda p: p.name.lower())

    def _cleanup_temp_files(self) -> None:
        for file_path in self.input_dir.glob(f"*_temp{TARGET_FORMAT}"):
            try:
                file_path.unlink()
            except OSError:
                pass

    def _validate_files(self, all_files: List[Path]) -> Tuple[List[PreparedFile], List[Path]]:
        console.print(f"\n| Fase 1: Validando {len(all_files)} archivos...", style=STYLE_INFO)
        prepared_files: List[PreparedFile] = []
        files_needing_conversion: List[Path] = []
        invalid_messages: List[str] = []
        warning_messages: List[str] = []
        max_size = self.provider.max_file_size_mb()

        with make_progress("Validando...") as progress:
            task = progress.add_task("", total=len(all_files))
            for file_path in all_files:
                validation = self.audio_validator(file_path, max_size, self.ffmpeg_available)

                if validation.warning_message:
                    warning_messages.append(f"|   - {validation.warning_message}")

                if validation.is_valid:
                    if validation.needs_conversion:
                        files_needing_conversion.append(file_path)
                    else:
                        prepared_files.append(PreparedFile(file_path, file_path))
                else:
                    invalid_messages.append(f"|   - {file_path.name}: {validation.error_message}")
                progress.advance(task)

        if warning_messages:
            console.print(f"\n| {style_info('Advertencias de validación:')}", style=STYLE_INFO)
            for msg in warning_messages:
                console.print(msg, style=STYLE_INFO)

        if invalid_messages:
            console.print(f"\n| {style_error('Archivos omitidos:')}", style=STYLE_ERROR)
            for msg in invalid_messages:
                console.print(msg, style=STYLE_ERROR)

        return prepared_files, files_needing_conversion

    def _convert_files(self, files_needing_conversion: List[Path]) -> List[PreparedFile]:
        if not files_needing_conversion:
            console.print("\n| Fase 2: No se requiere conversión.", style=STYLE_INFO)
            return []

        console.print(
            f"\n| Fase 2: Convirtiendo {len(files_needing_conversion)} archivos a {TARGET_FORMAT}...",
            style=STYLE_INFO,
        )

        converted_files: List[PreparedFile] = []
        converted_count = 0
        failed_count = 0

        with make_progress("Convirtiendo...") as progress:
            task = progress.add_task("", total=len(files_needing_conversion))
            workers = min(os.cpu_count() or 4, MAX_CONVERSION_WORKERS)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.audio_converter, path): path
                    for path in files_needing_conversion
                }
                for future in concurrent.futures.as_completed(futures):
                    original = futures[future]
                    conv_path, conv_error = future.result()

                    if conv_path:
                        converted_files.append(
                            PreparedFile(
                                original_path=original,
                                transcribe_path=conv_path,
                            )
                        )
                        try:
                            original.unlink()
                            converted_count += 1
                        except Exception as e:
                            console.print(
                                f"| {style_error(f'Error al eliminar {original.name}:')} {e}",
                                style=STYLE_ERROR,
                            )
                    else:
                        converted_files.append(
                            PreparedFile(
                                original_path=original,
                                transcribe_path=None,
                                error_message=conv_error or "Error de conversión.",
                            )
                        )
                        failed_count += 1

                    progress.advance(task)

        if converted_count:
            console.print(f"| {style_success('Convertidos:')} {converted_count} archivos", style=STYLE_SUCCESS)
        if failed_count:
            console.print(f"| {style_error('Fallos:')} {failed_count} archivos", style=STYLE_ERROR)

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
        console.print(
            f"\n| Fase 3: Transcribiendo {len(valid_files)} archivos con {style_keyword(self.provider.name)}...",
            style=STYLE_INFO,
        )

        success_count = 0
        error_count = 0

        with make_progress(f"Transcribiendo con {self.provider.name}...") as progress:
            task = progress.add_task("", total=len(valid_files))

            for file in valid_files:
                if file.transcribe_path is None:
                    continue

                result = self.provider.transcribe(
                    audio_path=file.transcribe_path,
                    original_filename=file.original_path.name,
                    prompt=prompt,
                )

                if result.error is None:
                    move_error = self._move_processed_audio(file.transcribe_path)
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

    def _move_processed_audio(self, origin: Path | None) -> Optional[str]:
        if origin is None:
            return "ruta de audio no disponible"
        try:
            destination = self.output_dir / "audio" / origin.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.file_mover(str(origin), str(destination))
            return None
        except Exception as e:
            console.print(
                f"| {style_error(f'Error al mover {origin.name}:')} {e}",
                style=STYLE_ERROR,
            )
            return str(e)

    def _print_summary(self, success_count: int, error_count: int) -> None:
        if success_count:
            console.print(f"\n| {style_success('Transcripciones exitosas:')} {success_count} archivos", style=STYLE_SUCCESS)
        if error_count:
            console.print(f"| {style_error('Transcripciones fallidas:')} {error_count} archivos", style=STYLE_ERROR)
        console.print(
            f"\n| Proceso completado. Revisa '{self.output_dir / 'msg'}' para las transcripciones.",
            style=STYLE_SUCCESS,
        )
