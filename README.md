# SpeechToText

CLI local en Python para transcribir audio por lotes con Google Gemini y FFmpeg.

Lee los archivos de un directorio de entrada (cualquier archivo con una pista de audio, incluidos
los vídeos), los convierte a un formato estándar cuando hace falta, los envía a Gemini y guarda
cada transcripción como un archivo `.txt` en el directorio de salida.

## Estado del proyecto

Proyecto terminado y archivado: no recibe mantenimiento. Google retira modelos con frecuencia
(el que usaba la primera versión dejó de existir en mayo de 2026). Si algún modelo deja de
funcionar, revisa la [lista oficial](https://ai.google.dev/gemini-api/docs/models) y actualiza
`GEMINI_MODELS` en `config.py`. Los modelos incluidos son los vigentes en septiembre de 2026.

## Privacidad

Los audios y las transcripciones permanecen en carpetas locales ignoradas por Git. El audio se
envía a Google Gemini únicamente para ejecutar la transcripción solicitada: se sube a la Files
API de Google y se borra en cuanto termina (si el borrado fallara, Google lo elimina por su
cuenta a las 48 horas). El nombre del archivo no se envía.

## Requisitos

- Windows con Python 3.10 o superior en el `PATH` (el código no depende del sistema operativo,
  pero los lanzadores incluidos son para Windows).
- Una clave de API de Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
- [FFmpeg](https://ffmpeg.org/) (incluye `ffprobe`) en el `PATH`. Es lo que permite aceptar
  cualquier formato, extraer el audio de vídeos y dividir los audios muy largos. Sin FFmpeg solo
  se procesan los formatos que Gemini acepta directamente (MP3, M4A, AAC, OGG, OPUS, WEBM, WAV,
  FLAC y AIFF), sin conversión ni división.

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

| Modelo | Uso | `--prompt` | Máximo por petición |
| --- | --- | --- | --- |
| `gemini-3.5-transcribe` (por defecto) | Voz a texto dedicado: detecta el idioma y transcribe literalmente. | No | 30 min |
| `gemini-3.8-flash` | Modelo general más capaz; admite instrucciones en el prompt. | Sí | 30 min |
| `gemini-3.5-flash-lite` | Modelo general económico; admite instrucciones en el prompt. | Sí | 30 min |

Combinar `--prompt` con `gemini-3.5-transcribe` es un error: ese modelo solo recibe audio. Los
audios que superan el máximo por petición no se rechazan: se dividen en tramos (ver más abajo).

## Formatos, conversión y tamaño

- **Cualquier audio o vídeo.** Con FFmpeg, un archivo se considera audio por su contenido y no
  por su extensión: sirve cualquier archivo con pista de audio que FFmpeg sepa leer (MP3, WAV,
  FLAC, OGG, M4A, WMA, AMR, AC3, MKV, MP4, MOV...). En los vídeos se usa la primera pista de
  audio. Los archivos que no contienen audio (imágenes, textos...) se ignoran con un aviso y no
  cuentan como error.
- **Sin conversión cuando no hace falta.** MP3, M4A (AAC), AAC, OGG, OPUS y WEBM se envían tal
  cual.
- **Conversión rápida al resto.** WAV, AIFF, FLAC, WMA, vídeos y demás se convierten a MP3
  mono de 16 kHz y 48 kbps, el formato estándar para voz (Gemini ya mezcla los canales a mono y
  reduce la calidad internamente). Se codifica muy rápido, en paralelo si hay varios archivos,
  y reduce mucho el tamaño: un WAV de 100 MB queda en pocos MB.
- **Originales intactos.** La conversión se hace en un directorio temporal. Los audios originales
  nunca se modifican ni se borran: solo se mueven a `salida/audio` cuando su transcripción sale
  bien.
- **Tamaño.** Hasta 2 GB por archivo (límite de la Files API). Un archivo que lo supere se
  convierte antes de enviarlo, así que con FFmpeg en la práctica no hay límite por tamaño; sin
  FFmpeg se rechaza.
- **Duración.** Los modelos admiten audios de más de una hora, pero por encima de ~30 minutos la
  calidad cae (el modelo dedicado añade texto repetido al final y los `flash` se saltan partes).
  Por eso, si un audio supera los 30 minutos, se divide en tramos de hasta 27 minutos, cortando
  en las pausas del habla para no partir palabras. Los tramos se transcriben en orden y sus
  textos se unen en un único `.txt`. Si falla cualquiera, el archivo completo se marca como
  fallido y se puede reintentar.
- Con FFmpeg, los audios de menos de 1 segundo se descartan.
- Cada petición HTTP tiene un timeout de 15 minutos, y los errores transitorios del servidor
  (códigos 408, 500, 502, 503 y 504) se reintentan hasta 5 intentos en total, con espera creciente
  (5, 10, 20 y 40 s). Además, si la API responde con un límite de tasa por minuto (429), el
  programa espera el tiempo que ella indica (hasta 2 minutos) y reintenta hasta 3 veces; así los
  tramos de un audio largo, que se envían seguidos, no fallan por ese límite.

## Salidas

- Transcripciones: `salida/msg/<nombre-original>.<extensión>.txt`
- Audios procesados: `salida/audio/`, con su nombre y formato originales.
- Errores de validación: se muestran en consola.
- Errores de conversión o transcripción: se registran en el `.txt` correspondiente y el audio
  se queda en `entrada` para poder reintentarlo.

El nombre conserva la extensión original para evitar colisiones entre `clip.wav` y `clip.mp3`.
Si repites un nombre que ya se procesó, nada se sobrescribe: el audio nuevo se archiva como
`nombre.1.ext` y la transcripción como `nombre.<extensión>.1.txt`. La única excepción es el
informe de error de un intento anterior, que un reintento sí reemplaza.

## Pruebas

Las pruebas usan `unittest` y no llaman a Gemini (usan clientes simulados, incluido el SDK real
sobre un transporte HTTP falso). Las que ejercitan FFmpeg de verdad se omiten si no está
instalado.

```bat
python -m unittest discover -s tests
```

## Estructura

- `main.py`: CLI, carga `.env`, inicializa el proveedor y lanza el procesamiento.
- `config.py`: catálogo de modelos y sus límites, formatos que Gemini acepta directamente,
  parámetros de conversión y rutas por defecto.
- `providers.py`: proveedor Gemini y llamadas a `google-genai`.
- `audio.py`: detección de audio con `ffprobe`, validación, conversión a MP3 y división en tramos.
- `transcriber.py`: orquesta validación, conversión, transcripción y archivado de audios.
- `fileutils.py`: nombres de archivo únicos para no sobrescribir salidas.
- `ui.py`: salida por consola y escritura de las transcripciones.
- `models.py`: resultado de una transcripción.
- `run.bat`: lanzador para Windows que prepara el entorno la primera vez.
- `tests/`: pruebas de validación, conversión, proveedor, flujo completo y salidas.

## Licencia

[MIT](LICENSE).
