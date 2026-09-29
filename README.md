# SpeechToText

CLI local en Python para transcribir audio por lotes con Google Gemini y FFmpeg.

Toma los archivos de un directorio de entrada, valida formato, tamaño y duración, convierte a
MP3 cuando hace falta, envía el audio a Gemini y guarda cada transcripción como un archivo
`.txt` en el directorio de salida.

## Estado del proyecto

Proyecto terminado y archivado: no recibe mantenimiento. Google retira modelos con frecuencia
(el que usaba la primera versión dejó de existir en mayo de 2026). Si algún modelo deja de
funcionar, revisa la [lista oficial](https://ai.google.dev/gemini-api/docs/models) y actualiza
`GEMINI_MODELS` en `config.py`. Los modelos incluidos son los vigentes en septiembre de 2026.

## Privacidad

Los audios y las transcripciones permanecen en carpetas locales ignoradas por Git. El audio se
envía a Google Gemini únicamente para ejecutar la transcripción solicitada. Con el modelo por
defecto se sube a la Files API de Google y se borra en cuanto termina la transcripción (si el
borrado fallara, Google lo elimina por su cuenta a las 48 horas).

## Requisitos

- Windows con Python 3.10 o superior en el `PATH` (el código no depende del sistema operativo,
  pero los lanzadores incluidos son para Windows).
- Una clave de API de Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
- [FFmpeg](https://ffmpeg.org/) en el `PATH` para convertir formatos que no sean MP3 y para
  validar la duración. Sin FFmpeg solo se procesan MP3 y no se comprueban los límites de duración.

## Instalación y uso

Ejecuta `run.bat` (con doble clic o desde una terminal). La primera vez crea el entorno virtual
`venv`, instala las dependencias de `requirements.in` y crea `.env` a partir de `.env.example`.
Abre `.env` y reemplaza `tu_clave_aqui` por tu `GOOGLE_API_KEY`; luego copia los audios en
`entrada` y vuelve a ejecutar `run.bat`.

Instalación manual, si prefieres no usar el lanzador:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.in
copy .env.example .env
python main.py
```

`run.bat` acepta las mismas opciones que `main.py`:

```bat
run.bat --input entrada --output salida
run.bat --model gemini-3.8-flash --prompt "Transcribe con puntuación clara"
```

Por defecto lee de `entrada` y escribe en `salida`, relativos al directorio desde el que se
ejecuta (`run.bat` siempre trabaja desde la carpeta del proyecto). Ambas se crean solas y están
ignoradas por Git. El programa termina con código de salida 1 si algún archivo falla o se omite
por no ser válido, y con 0 si todo salió bien.

## Modelos

El proveedor activo es `gemini`. Se elige con `--model`:

| Modelo | Uso | `--prompt` | Límites por archivo |
| --- | --- | --- | --- |
| `gemini-3.5-transcribe` (por defecto) | Voz a texto dedicado: detecta el idioma y transcribe literalmente. El audio se sube por la Files API. | No | 1 hora; 2 GB (límite de la Files API) |
| `gemini-3.8-flash` | Modelo general más capaz; admite instrucciones en el prompt. Audio en línea. | Sí | 20 MB |
| `gemini-3.5-flash-lite` | Modelo general económico; admite instrucciones en el prompt. Audio en línea. | Sí | 20 MB |

Combinar `--prompt` con `gemini-3.5-transcribe` es un error: ese modelo solo recibe audio.

## Formatos y límites

Formatos aceptados:

```text
.mp3, .mp4, .m4a, .wav, .flac, .ogg, .aac, .wma, .opus, .webm,
.aiff, .mpeg, .mpga
```

- Todo lo que no sea MP3 requiere FFmpeg y se convierte a `.mp3` (128 kbps) antes de
  transcribirse. **Cuando la conversión termina bien, el audio original se elimina.**
- Los límites de tamaño y duración se validan antes de convertir, sobre el archivo original. Un
  WAV grande puede superar el límite de 20 MB de los modelos `flash` aunque su MP3 cupiera;
  con el modelo por defecto (2 GB) rara vez importa.
- Con FFmpeg, los audios de menos de 1 segundo se descartan.
- Cada petición HTTP tiene un timeout de 15 minutos, y los errores transitorios (códigos 408,
  429, 500, 502, 503 y 504) se reintentan hasta 3 intentos en total, con espera creciente.

## Salidas

- Transcripciones: `salida/msg/<nombre-original>.<extensión>.txt`
- Audios procesados: `salida/audio/`, con la extensión original más `.mp3` si hubo conversión
  (`clip.wav` queda como `clip.wav.mp3`).
- Errores de validación: se muestran en consola.
- Errores de conversión o transcripción: se registran en el `.txt` correspondiente y el audio
  se queda en `entrada` para poder reintentarlo.

El nombre conserva la extensión original para evitar colisiones entre `clip.wav` y `clip.mp3`.
Si repites un nombre que ya se procesó, nada se sobrescribe: el audio nuevo se archiva como
`nombre.1.mp3` y la transcripción como `nombre.<extensión>.1.txt`. La única excepción es el
informe de error de un intento anterior, que un reintento sí reemplaza.

## Pruebas

Las pruebas usan `unittest` y no llaman a Gemini (usan clientes simulados, incluido el SDK real
sobre un transporte HTTP falso).

```bat
python -m unittest discover -s tests
```

## Estructura

- `main.py`: CLI, carga `.env`, inicializa el proveedor y lanza el procesamiento.
- `config.py`: catálogo de modelos y sus capacidades, límites, extensiones aceptadas y rutas por
  defecto.
- `providers.py`: proveedor Gemini y llamadas a `google-genai`.
- `audio.py`: validación, detección de FFmpeg y conversión a MP3 (los parciales usan el sufijo
  `.stt-tmp`).
- `transcriber.py`: orquesta validación, conversión, transcripción y archivado de audios.
- `fileutils.py`: nombres de archivo únicos para no sobrescribir salidas.
- `ui.py`: salida por consola y escritura de las transcripciones.
- `models.py`: resultado de una transcripción.
- `run.bat`: lanzador para Windows que prepara el entorno la primera vez.
- `tests/`: pruebas de validación, conversión, proveedor, flujo completo y salidas.
