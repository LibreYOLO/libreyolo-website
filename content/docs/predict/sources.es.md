---
title: Fuentes de predicción
seo_title: "Fuentes de predicción en LibreYOLO"
description: "Todas las fuentes que acepta predict: imágenes, carpetas, URLs, archivos de vídeo, webcams, RTSP, YouTube, captura de pantalla, listas de imágenes y archivos .streams."
lead: "El argumento source se clasifica antes de abrir nada, así que una sola llamada resuelve un JPEG, una carpeta, un MP4, un índice de webcam, una URL RTSP, una región de la pantalla o una lista de cámaras."
keywords:
  - inferencia de vídeo yolo python
  - rtsp
  - detección de objetos webcam python
  - predecir sobre una carpeta de imágenes
  - detección de objetos en captura de pantalla
  - varios streams rtsp
  - archivo streams
  - inferencia youtube
  - vid_stride
  - stream=True
last_verified: "1.5.0"
verification: "Clasificación de fuentes leída de libreyolo/utils/source.py (classify_source, SourceKind, StreamSource, MultiStreamSource). Tipos de imagen aceptados y extensiones de directorio de libreyolo/utils/image_loader.py. Extensiones de vídeo y rutas de guardado de libreyolo/utils/video.py. Sintaxis de screen de libreyolo/utils/screen.py. Formas de retorno y valores por defecto de los argumentos de InferenceRunner.__call__ en libreyolo/models/base/inference.py."
snippets:
  images:
    - label: Una imagen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Una fuente de una sola imagen devuelve un Results, no una lista.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Imágenes en memoria
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: Una carpeta
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("sample_folder")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # Una carpeta devuelve una lista, un Results por imagen, ordenada por ruta.
        results = model(str(folder))
        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Un archivo de vídeo (usa tu propio clip)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sustituye clip.mp4 por un archivo de vídeo que tengas en disco.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: Un fotograma de cada tres, escrito en disco
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webcam (requiere una cámara conectada)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Índice de webcam 0. Las fuentes en directo no terminan nunca, así que acota el bucle.
        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (requiere una URL de cámara accesible)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Un archivo .streams (usa tus propias cámaras)
      language: python
      code: |
        import itertools
        from pathlib import Path

        from libreyolo import LibreYOLO

        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )

        model = LibreYOLO("LibreYOLO9s.pt")
        for result in itertools.islice(model("cameras.streams", stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: Una lista de cámaras
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Una captura de pantalla (requiere mss y una sesión de escritorio)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sin stream=True esto captura un único fotograma.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: Una región de un monitor, en continuo
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "screen <monitor> <left> <top> <width> <height>"
        for result in itertools.islice(model("screen 1 100 200 512 256", stream=True), 50):
            print(len(result.boxes))
---

## Cómo se clasifica una fuente

`classify_source` inspecciona el valor antes de que se abra o se descargue nada,
en este orden. Gana la primera regla que coincide.

| Fuente | Se interpreta como |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Captura de pantalla |
| Un `int` no negativo, o una cadena de dígitos sin ningún archivo con ese nombre | Webcam |
| Una URL `rtsp://`, `rtmp://`, `tcp://` o `udp://` | Stream de red |
| Una URL `http(s)://` cuya ruta termina en `.m3u8` | Stream de red |
| Una URL de página de YouTube | Stream de red |
| Una lista o tupla cuyas entradas son todas en directo o vídeo | Varios streams en directo |
| Cualquier otra lista o tupla | Batch de imágenes |
| Una ruta que termina en `.streams` | Varios streams en directo |
| Una ruta con extensión de vídeo | Archivo de vídeo |
| Un directorio existente | Carpeta de imágenes |
| Cualquier otra cosa | Una sola imagen |

Una lista que mezcla fuentes en directo con imágenes lanza `TypeError`. Un
índice de webcam negativo lanza `ValueError`.

El clasificador nunca toca la red, así que una URL mal escrita se manifiesta
cuando se abre la captura, no cuando se llama a `predict`.

## Imágenes

<code-tabs name="images" />

Una fuente de una sola imagen acepta siete tipos.

| Tipo | Se interpreta como |
|---|---|
| `str` o `pathlib.Path` | Archivo local, `http(s)://`, `s3://` o `gs://` |
| `PIL.Image.Image` | Convertida a RGB |
| `numpy.ndarray` | Escala de grises 2D, o 3D HWC o CHW; un array 4D usa su primera imagen |
| `torch.Tensor` | CHW o NCHW, leído como RGB; un tensor con batch usa su primera imagen |
| `bytes` | Datos de imagen codificados |
| `io.BytesIO` | Datos de imagen codificados |

Todo se convierte a RGB antes del preprocesado. Los arrays de NumPy son el único
caso en el que el orden de los canales es ambiguo, así que lo controla
`color_format`: `"auto"` (el valor por defecto) deja el array tal cual, `"bgr"`
invierte los canales, que es lo que necesita un fotograma leído con OpenCV.

Los arrays de coma flotante se reescalan según su propio rango: los valores
iguales o inferiores a `1.0` se multiplican por 255, y los valores más altos se
recortan al rango `[0, 255]`. Un array RGBA descarta su canal alfa.

Las rutas remotas necesitan un paquete cada una, y ninguno se instala por
defecto: `requests` para `http(s)://`, `boto3` para `s3://` y `gcsfs` para
`gs://`.

## Carpetas

Un directorio se recorre de forma recursiva y se ordena, y todo archivo con uno
de estos sufijos se convierte en una imagen: `.jpg`, `.jpeg`, `.png`, `.gif`,
`.webp`, `.bmp`, `.tiff`, `.tif`. Cualquier otra cosa dentro de la carpeta se
omite. Una carpeta vacía devuelve una lista vacía en lugar de lanzar un error.

Las carpetas y las listas son las dos fuentes que aceptan `batch`, que ejecuta
una pasada hacia delante apilada por bloque en las familias que lo soportan.
Consulta [Rendimiento de la inferencia](/docs/predict/performance).

## Archivos de vídeo

<code-tabs name="video" />

Una ruta cuenta como vídeo cuando su sufijo es uno de `.asf`, `.avi`, `.gif`,
`.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm`.

`.gif` aparece en las dos listas. Una ruta `.gif` pasada directamente a
`predict` se abre como vídeo, porque la comprobación de vídeo se ejecuta
primero; un `.gif` que está dentro de una carpeta escaneada se carga como imagen
fija.

`vid_stride` procesa uno de cada N fotogramas y su valor por defecto es `1`. Sin
`stream=True` el vídeo entero se decodifica en una lista, y cualquier cifra por
encima de 500 fotogramas tras aplicar el salto emite un aviso que sugiere
`stream=True`.

Cada `Results` procedente de un vídeo lleva `frame_idx`.

## Webcams, streams de red y YouTube

<code-tabs name="live" />

Las fuentes en directo no tienen fin, así que requieren `stream=True`. Sin él,
`predict` lanza `ValueError` en vez de intentar recopilar una lista infinita.

Los fotogramas se leen en un hilo en segundo plano, uno por captura. Por defecto
la cola guarda solo el fotograma más reciente, así que un modelo más lento que
la cámara se salta fotogramas en lugar de quedarse atrás. `stream_buffer=True`
conserva todos los fotogramas capturados, lo que los preserva a costa de una
latencia creciente.

Un índice de webcam es un `int` o una cadena de dígitos. En Windows la captura
se abre primero a través del backend DirectShow y recurre al backend por defecto
si eso falla.

Las URLs de páginas de YouTube se resuelven a una URL de medios directa sin
descargar el vídeo, para lo que hace falta `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

Las etiquetas de stream se censuran antes de registrarse en el log o de usarse
como nombres de archivo. Una URL que lleva credenciales aparece como
`user:***@host`, y las cadenas de consulta se eliminan de las etiquetas de
stream directas porque ahí es donde viven las URLs firmadas y los tokens bearer.
El id de un vídeo de YouTube se conserva, ya que no es una credencial.

## Varias cámaras a la vez

<code-tabs name="streams" />

Un archivo `.streams` es una fuente por línea. Las líneas en blanco y las que
empiezan por `#` se ignoran. Cada línea restante debe ser a su vez un índice de
webcam, un stream de red, una URL de YouTube o la ruta a un archivo de vídeo;
cualquier otra cosa lanza `ValueError` indicando el número de línea. Un archivo
vacío lanza un error en lugar de arrancar sin cámaras.

Una lista o tupla de fuentes en directo hace lo mismo sin necesidad de archivo.

Cada captura tiene su propio hilo, y los fotogramas de todas ellas se
multiplexan en un único generador. Cada pasada consulta todos los streams
activos y entrega lo que esté listo, así que una cámara lenta no frena a una
rápida, y los fotogramas de distintas cámaras se intercalan. Un stream que
termina sale de la rotación mientras los demás continúan.

## Captura de pantalla

<code-tabs name="screen" />

Una fuente de pantalla es la palabra `screen` seguida de cero, uno, cuatro o
cinco enteros. Cualquier otra cantidad lanza `ValueError`.

| Forma | Captura |
|---|---|
| `"screen"` | Todos los monitores, fusionados |
| `"screen 1"` | El monitor 1 |
| `"screen 100 200 512 256"` | Un recuadro sobre el escritorio fusionado |
| `"screen 1 100 200 512 256"` | Un recuadro sobre el monitor 1 |

Las coordenadas del recuadro son `left top width height`, relativas a la esquina
superior izquierda del monitor elegido. Una fuente de pantalla informa de sus
FPS como 30 dividido entre `vid_stride`, que es la tasa a la que se escribe un
vídeo guardado. La captura necesita el paquete `mss`:

```bash
pip install mss
```

Sin `stream=True`, una fuente de pantalla captura un fotograma y devuelve un
único `Results`, que es el equivalente en captura de pantalla a predecir sobre
un archivo de imagen. Con `stream=True` captura hasta que se rompe el bucle.

## Qué devuelve predict

La forma del valor de retorno depende de la fuente y de `stream`.

| Fuente | `stream=False` | `stream=True` |
|---|---|---|
| Una sola imagen | Un `Results` | Generador de un `Results` |
| Lista de imágenes | Lista de `Results` | Generador |
| Carpeta | Lista de `Results` | Generador |
| Archivo de vídeo | Lista de `Results` | Generador |
| Pantalla | Un `Results` | Generador, sin fin |
| Webcam, stream de red, `.streams` | `ValueError` | Generador, sin fin |

Una sola imagen devuelve el propio objeto `Results`. Indexarlo selecciona una
detección, no una imagen, así que `result[0]` en una predicción sobre una sola
imagen es el primer box y no la primera foto. Para saber qué llevan esos
objetos, consulta [Trabajar con resultados](/docs/predict/results).

## Dónde escribe save

`save=True` escribe la salida anotada en un directorio de ejecución en lugar de
devolverla.

Las imágenes van a un `runs/detect/predict`, `runs/detect/predict2` y así
sucesivamente que se autoincrementa, conservando el nombre de archivo de origen.
Todas las imágenes de un mismo proceso acaban en el mismo directorio, así que
dos carpetas de entrada que contengan el mismo nombre de archivo se sobrescriben
entre sí. Las imágenes en memoria no tienen nombre de archivo que reutilizar y
se numeran `image0`, `image1` y así sucesivamente.

Las fuentes de vídeo y en directo se escriben como un único `.mp4` con el nombre
de la fuente.

`output_path` anula el directorio. Una ruta con sufijo se trata como un archivo,
y una ruta sin él como un directorio. `output_file_format` selecciona la
codificación de las imágenes fijas y acepta `jpg`, `png` o `webp`.

Tras un guardado, la ruta escrita también se adjunta al resultado como
`result.saved_path`.
