"""Transcribe los audios de entrada/ y guarda cada transcripción en salida/.

    python -m speech_to_text                              modelo por defecto
    python -m speech_to_text --model gemini-3.8-flash --prompt "..."
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from .config import (
    AVAILABLE_GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    GEMINI_MODELS,
)
from .providers import get_provider
from .transcriber import Transcriber
from .ui import console, style_error, style_keyword, STYLE_INFO, STYLE_ERROR, STYLE_DEFAULT, STYLE_SUCCESS


def make_output_safe() -> None:
    """Evita que un carácter no representable (p. ej. un nombre de archivo en japonés con la
    salida redirigida a cp1252) lance UnicodeEncodeError y aborte todo el lote."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def build_parser() -> argparse.ArgumentParser:
    prompt_models = ", ".join(m.id for m in GEMINI_MODELS.values() if m.accepts_prompt)

    parser = argparse.ArgumentParser(
        prog="python -m speech_to_text",
        description="Transcribe archivos de audio usando Google Gemini"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        choices=AVAILABLE_GEMINI_MODELS,
        help=f"Modelo de transcripción (default: {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help=f"Prompt para mejorar la transcripción (solo con: {prompt_models})",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directorio de entrada (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main() -> int:
    make_output_safe()
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    if args.prompt and not GEMINI_MODELS[args.model].accepts_prompt:
        parser.error(
            f"{args.model} es un modelo de transcripción dedicado y no admite --prompt. "
            f"Quita --prompt o elige otro modelo con --model."
        )

    console.print("| Iniciando SpeechToText", style=STYLE_INFO)

    model_list = ", ".join(AVAILABLE_GEMINI_MODELS)
    console.print(
        f"| Proveedor: {style_keyword('gemini')} | Modelos: {model_list}",
        style=STYLE_DEFAULT,
    )

    try:
        provider = get_provider(model=args.model)
        provider.initialize()
        console.print(f"| Proveedor inicializado: {style_keyword(provider.name)}", style=STYLE_SUCCESS)
        console.print(f"| Modelo activo: {style_keyword(provider.current_model())}", style=STYLE_DEFAULT)
    except ValueError as e:
        console.print(f"\n| {style_error('Error:')} {e}\n", style=STYLE_ERROR)
        return 1

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = Transcriber(
        provider=provider,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    failed_count = transcriber.process_files(prompt=args.prompt)

    console.print("\n| Finalizado\n", style=STYLE_INFO)
    return 1 if failed_count else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.print("\n| Interrumpido por el usuario.\n", style=STYLE_INFO)
        raise SystemExit(130)
    except Exception as e:
        console.print(f"\n| {style_error('Error crítico:')} {e}\n", style=STYLE_ERROR)
        if os.getenv("SPEECH_TO_TEXT_DEBUG") == "1":
            traceback.print_exc()
        raise SystemExit(1)
