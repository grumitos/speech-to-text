# SpeechToText

CLI local para transcribir archivos de audio con Google Gemini.

## Estado del proyecto

**En desarrollo activo.** El flujo por lotes, la conversion de audio y la suite de pruebas son funcionales. La integracion de IA es intencionalmente acotada a Gemini y utiliza un modelo preview configurable, por lo que puede requerir ajustes cuando cambie la disponibilidad del proveedor.

El flujo actual toma archivos desde un directorio de entrada, valida formato/tamano,
convierte a MP3 cuando hace falta, envia el audio al modelo Gemini configurado y guarda
las transcripciones como archivos `.txt` en el directorio de salida.

Los audios y transcripciones permanecen en directorios locales ignorados por Git. El audio se envia a Google Gemini unicamente para ejecutar la transcripcion solicitada.

## Requisitos

- Python 3.10 o superior.
- Una clave `GOOGLE_API_KEY` en `.env`.
- FFmpeg instalado y disponible en `PATH` para convertir formatos que no sean MP3.
  Si no esta disponible, los MP3 pueden procesarse, pero se omite la validacion de duracion.

## Configuracion

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.in
copy .env.example .env
```

Despues edita `.env` y reemplaza `GOOGLE_API_KEY=tu_clave_aqui` por una clave valida.
No subas `.env` al repo; esta ignorado por Git.

## Uso

Por defecto, el programa lee audio desde `entrada` y escribe resultados en `salida`.
Ambos directorios se crean automaticamente si no existen y estan ignorados por Git.

```bat
python main.py
```

Tambien puedes usar el wrapper de Windows:

```bat
run.bat
```

Opciones disponibles:

```bat
python main.py --input entrada --output salida --prompt "Transcribe con puntuacion clara"
python main.py --model gemini-3.1-flash-lite-preview
```

## Modelo y formatos

El proveedor activo es `gemini` y el unico modelo soportado por la configuracion actual es
`gemini-3.1-flash-lite-preview`.

Formatos de audio aceptados:

```text
.mp3, .mp4, .m4a, .wav, .flac, .ogg, .aac, .wma, .opus, .webm,
.aiff, .mpeg, .mpga
```

Los archivos que no sean MP3 requieren FFmpeg y se convierten a `.mp3` antes de
transcribirse. Cuando una conversion termina correctamente, el audio original se elimina.
Tras una transcripcion exitosa, el audio usado para transcribir se mueve a `salida/audio`.

## Salidas

- Transcripciones: `salida/msg/<nombre-original>.<extension>.txt`
- Audios procesados: `salida/audio/`
- Errores de validacion: se muestran en consola.
- Errores de conversion o transcripcion: se registran en el `.txt` correspondiente.

El nombre de salida conserva la extension original del archivo de entrada para evitar
colisiones como `clip.wav` y `clip.mp3`.

## Pruebas

Las pruebas usan `unittest` y no requieren llamar a Gemini.

```bat
python -m unittest discover -s tests
```

## Estructura principal

- `main.py`: CLI, carga `.env`, inicializa el proveedor y lanza el procesamiento.
- `config.py`: modelo activo, limites, extensiones aceptadas y rutas por defecto.
- `providers.py`: proveedor Gemini y llamada a `google-genai`.
- `audio.py`: validacion, deteccion de FFmpeg y conversion a MP3.
- `transcriber.py`: orquestacion de validacion, conversion, transcripcion y movimiento de archivos.
- `ui.py`: salida por consola y escritura de archivos de transcripcion.
- `tests/`: cobertura de validacion, proveedor, flujo de transcripcion y escritura de salidas.
