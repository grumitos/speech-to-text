import textwrap
from pathlib import Path

try:
    from rich.console import Console
    from rich.markup import escape as rich_escape
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    RICH_AVAILABLE = True
except ModuleNotFoundError:
    RICH_AVAILABLE = False

    class Console:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def print(self, *args, **kwargs) -> None:
            print(*args)

    class TextColumn:
        def __init__(self, template: str, *args, **kwargs):
            self.template = template

    class BarColumn:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class MofNCompleteColumn:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class SpinnerColumn:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class TimeElapsedColumn:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class Progress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self) -> "Progress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def add_task(self, description: str, total: int = 0) -> int:
            return 0

        def advance(self, task_id: int, advance: int = 1) -> None:
            return None

from models import TranscriptionResult

LIGHT_TOKENS = {
    "canvas": "#f8f8f6",
    "text": "#121212",
    "surface": "#ffffff",
    "secondary": "#efeeeb",
    "muted": "#7b7974",
    "border": "#1f1f1e26",
    "border_solid": "#d9d8d5",
    "accent": "#d97757",
    "focus": "#2977d6",
}

DARK_TOKENS = {
    "canvas": "#1f1f1e",
    "text": "#f8f8f6",
    "surface": "#121212",
    "surface_elevated": "#2c2c2a",
    "muted": "#97958c",
    "border": "#e2e1da26",
    "border_solid": "#333331",
    "accent": "#d97757",
    "focus": "#3886e5",
}

UI_MODE = "dark"
UI_TOKENS = DARK_TOKENS

STYLE_DEFAULT = UI_TOKENS["text"]
STYLE_KEYWORD = f"bold {UI_TOKENS['accent']}"
STYLE_SUCCESS = "bold #74a47f"
STYLE_ERROR = "bold #d76b63"
STYLE_WARNING = "bold #c99746"
STYLE_INFO = UI_TOKENS["focus"]
STYLE_MUTED = UI_TOKENS["muted"]
STYLE_DISABLED = f"dim {UI_TOKENS['muted']}"
STYLE_BORDER = UI_TOKENS["border_solid"]

STATE_LABELS = {
    "info": "INFO",
    "success": "OK",
    "warning": "AVISO",
    "error": "ERROR",
    "empty": "VACIO",
    "disabled": "OFF",
    "loading": "RUN",
}

STATE_STYLES = {
    "info": STYLE_INFO,
    "success": STYLE_SUCCESS,
    "warning": STYLE_WARNING,
    "error": STYLE_ERROR,
    "empty": STYLE_MUTED,
    "disabled": STYLE_DISABLED,
    "loading": STYLE_KEYWORD,
}

console = Console(highlight=False)


def escape_markup(text: object) -> str:
    raw = str(text)
    if not RICH_AVAILABLE:
        return raw
    return rich_escape(raw)


def style_text(text: object, style: str) -> str:
    if not RICH_AVAILABLE:
        return str(text)
    return f"[{style}]{escape_markup(text)}[/]"


def style_keyword(text: str) -> str:
    return style_text(text, STYLE_KEYWORD)


def style_success(text: str) -> str:
    return style_text(text, STYLE_SUCCESS)


def style_error(text: str) -> str:
    return style_text(text, STYLE_ERROR)


def style_info(text: str) -> str:
    return style_text(text, STYLE_INFO)


def style_warning(text: str) -> str:
    return style_text(text, STYLE_WARNING)


def style_muted(text: str) -> str:
    return style_text(text, STYLE_MUTED)


def _state_badge(state: str) -> str:
    label = STATE_LABELS.get(state, state.upper())
    style = STATE_STYLES.get(state, STYLE_INFO)
    return style_text(f"[{label}]", style)


def print_section(title: str) -> None:
    console.print(f"\n| {style_text(title, STYLE_KEYWORD)}", style=STYLE_DEFAULT)


def print_state(state: str, message: object) -> None:
    console.print(
        f"| {_state_badge(state)} {escape_markup(message)}",
        style=STYLE_DEFAULT,
    )


def print_kv(label: str, value: object, value_style: str = STYLE_DEFAULT) -> None:
    console.print(
        f"| {style_muted(label + ':')} {style_text(value, value_style)}",
        style=STYLE_DEFAULT,
    )


def print_bullet(message: object, style: str = STYLE_DEFAULT) -> None:
    console.print(f"|   - {escape_markup(message)}", style=style)


def make_progress(description: str) -> Progress:
    return Progress(
        SpinnerColumn(style=STYLE_KEYWORD),
        TextColumn(description, style=STYLE_MUTED),
        BarColumn(
            bar_width=24,
            complete_style=UI_TOKENS["accent"],
            finished_style=STYLE_SUCCESS,
            pulse_style=STYLE_INFO,
        ),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def write_transcription_file(result: TranscriptionResult, output_dir: Path) -> None:
    msg_dir = output_dir / "msg"
    msg_dir.mkdir(parents=True, exist_ok=True)
    output_file = msg_dir / f"{Path(result.file_name).name}.txt"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            header = f" {result.file_name} "
            f.write(f"{'=' * 30} {header}{'=' * 30}\n")
            f.write(f"Fecha: {result.date}\n")
            f.write(f"Modelo: {result.model_name} ({result.provider_name})\n")
            if result.postprocess_warning:
                f.write(f"ADVERTENCIA: {result.postprocess_warning}\n")

            if result.conversion_error:
                f.write(f"ERROR DE CONVERSION: {result.conversion_error}\n\n")
            elif result.error:
                f.write(f"ERROR DE TRANSCRIPCION: {result.error}\n\n")
            else:
                f.write("\nTRANSCRIPCION:\n")
                f.write("-" * 85 + "\n")
                f.write(textwrap.fill(result.transcription_text, width=80) + "\n")
                f.write("-" * 85 + "\n")
            f.write("\n\n")
        print_state("success", f"Transcripcion guardada: {output_file}")
    except Exception as e:
        print_state("error", f"Error al escribir en {output_file}: {e}")
