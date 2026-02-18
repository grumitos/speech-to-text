import textwrap
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from models import TranscriptionResult

# Consola global
console = Console()

# Estilos
STYLE_DEFAULT = "white"
STYLE_KEYWORD = "bright_magenta"
STYLE_SUCCESS = "bright_green"
STYLE_ERROR = "red"
STYLE_INFO = "cyan"


def style_keyword(text: str) -> str:
    return f"[{STYLE_KEYWORD}]{text}[/{STYLE_KEYWORD}]"


def style_success(text: str) -> str:
    return f"[{STYLE_SUCCESS}]{text}[/{STYLE_SUCCESS}]"


def style_error(text: str) -> str:
    return f"[{STYLE_ERROR}]{text}[/{STYLE_ERROR}]"


def style_info(text: str) -> str:
    return f"[{STYLE_INFO}]{text}[/{STYLE_INFO}]"


def make_progress(description: str) -> Progress:
    return Progress(
        TextColumn(description),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    )


def write_transcription_file(result: TranscriptionResult, output_dir: Path) -> None:
    msg_dir = output_dir / "msg"
    msg_dir.mkdir(parents=True, exist_ok=True)
    output_file = msg_dir / f"{Path(result.file_name).stem}.txt"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            header = f" {result.file_name} "
            f.write(f"{'=' * 30} 📁{header}{'=' * 30}\n")
            f.write(f"📅 Fecha: {result.date}\n")
            f.write(f"🤖 Modelo: {result.model_name} ({result.provider_name})\n")

            if result.conversion_error:
                f.write(f"❌ ERROR DE CONVERSIÓN: {result.conversion_error}\n\n")
            elif result.error:
                f.write(f"❌ ERROR DE TRANSCRIPCIÓN: {result.error}\n\n")
            else:
                f.write(f"\n📝 TRANSCRIPCIÓN:\n")
                f.write("-" * 85 + "\n")
                f.write(textwrap.fill(result.transcription_text, width=80) + "\n")
                f.write("-" * 85 + "\n")
            f.write("\n\n")
        console.print(f"| Transcripción guardada en: {output_file}", style=STYLE_SUCCESS)
    except Exception as e:
        console.print(f"| {style_error(f'Error al escribir en {output_file}:')} {e}", style=STYLE_ERROR)
