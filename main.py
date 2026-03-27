from pathlib import Path

from dotenv import load_dotenv

from config import (
    AVAILABLE_GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
)
from providers import get_provider
from transcriber import Transcriber
from ui import console, style_error, style_keyword, STYLE_INFO, STYLE_ERROR, STYLE_DEFAULT, STYLE_SUCCESS


def main():
    import warnings
    import argparse

    warnings.filterwarnings("ignore", category=UserWarning, module=".*")
    load_dotenv()

    parser = argparse.ArgumentParser(
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
        help="Prompt para mejorar la transcripción",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json"],
        help="Formato de respuesta (default: text)",
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
    args = parser.parse_args()

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
        return

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = Transcriber(
        provider=provider,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    transcriber.process_files(
        prompt=args.prompt,
        response_format=args.format,
    )

    console.print("\n| Finalizado\n", style=STYLE_INFO)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n| Interrumpido por el usuario.\n", style=STYLE_INFO)
    except Exception as e:
        console.print(f"\n| {style_error('Error crítico:')} {e}\n", style=STYLE_ERROR)
        import traceback
        traceback.print_exc()
