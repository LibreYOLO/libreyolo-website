---
title: libreyolo predict
seo_title: referencia del comando libreyolo predict
description: >-
  Ejecuta inferencia desde la línea de comandos: cada argumento, su valor por
  defecto leído de la definición de la CLI y los flags que cambian lo que llega
  a stdout.
lead: >-
  Ejecuta un modelo cargado sobre una fuente e imprime las predicciones. La
  fuente puede ser una imagen, un directorio, un video, una URL o un stream en
  directo; el modelo puede ser un checkpoint o un artefacto exportado.
keywords:
  - libreyolo predict cli
  - inferencia yolo linea de comandos
  - comando predict libreyolo
  - argumentos libreyolo predict
  - yolo salida json terminal
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo predict
    mono: true
  - label: Requerido
    value: source
    mono: true
  - label: Salida
    value: >-
      Predicciones por stdout. Con save=true, archivos anotados en
      runs/detect/predict
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Guardar imágenes anotadas
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Clases filtradas, JSON por stdout'
      language: bash
      code: >
        # la clase 0 es person en la lista de clases COCO que acompaña al
        checkpoint.

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## Sinopsis

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Los argumentos son pares `key=value`. El mismo comando acepta también la forma
POSIX, de modo que `conf=0.4` y `--conf 0.4` son intercambiables, y un booleano
escrito `save=true` se convierte en `--save`. Los nombres con guion bajo
aceptan ambas grafías: `max_det=50` y `--max-det 50` llegan a la misma opción.

`libreyolo detect predict ...` se acepta y se comporta de forma idéntica; la
palabra de tarea se descarta antes del análisis.

## Argumentos

| Argumento | Por defecto | Significado |
|---|---|---|
| `source` | | Ruta de imagen, directorio o URL. Requerido |
| `model` | `yolox-s` | Nombre o ruta del modelo |
| `conf` | `0.25` | Umbral de confianza |
| `iou` | `0.45` | Umbral de IoU para NMS |
| `imgsz` | | Tamaño de la imagen de entrada: `640` (cuadrada) o `480x640` (alto x ancho). El tamaño de entrada propio del modelo si no se indica |
| `classes` | | Filtra por IDs de clase, p. ej. `[0,2,5]`. Se acepta un entero suelto |
| `max_det` | `300` | Máximo de detecciones por imagen |
| `half` | `false` | Inferencia en FP16 (solo CUDA, requiere soporte del modelo) |
| `save` | `false` | Guarda las imágenes anotadas |
| `batch` | `1` | Imágenes por pasada para fuentes de tipo directorio. Por encima de 1 ejecuta inferencia en batch real en los modelos que la soportan |
| `stream` | `false` | Devuelve los resultados de forma incremental. Se activa automáticamente para webcams y streams en directo |
| `stream_buffer` | `false` | Almacena en búfer cada frame en directo en lugar de conservar solo el más reciente |
| `vid_stride` | `1` | Procesa uno de cada N frames de video o de directo |
| `show` | `false` | Muestra los resultados de video y en directo; `q` detiene |
| `tiling` | `false` | Inferencia en mosaico (tiling) para imágenes grandes |
| `overlap_ratio` | `0.2` | Proporción de solapamiento entre mosaicos |
| `output_path` | | Ruta de salida explícita. En caso contrario, `project/name` cuando `save=true` |
| `color_format` | `auto` | Color de entrada: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Formato de salida: `jpg`, `png`, `webp` |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Modelo detector de caras (ruta o nombre de CLI). Requerido para los modelos de gaze |
| `gallery` | | Galería de caras `.npz` de `libreyolo enroll` contra la que identificar caras. Solo para modelos de embeddings faciales |
| `gallery_threshold` | `0.4` | Umbral de coseno para una coincidencia de identidad en la galería |
| `project` | `runs/detect` | Raíz del directorio de salida |
| `name` | `predict` | Nombre del experimento |
| `exist_ok` | `false` | Reutiliza el directorio de salida existente |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |
| `verbose` | `false` | Salida detallada por stderr |
| `help_json` | `false` | Vuelca el esquema del comando como JSON y sale |

## Ejemplos

<code-tabs name="examples" />

## Notas

Un artefacto exportado se carga igual que un checkpoint, por lo que
`model=weights/LibreYOLO9s.onnx` y `model=weights/LibreYOLO9s.engine` son
valores válidos para `model`. Tres opciones se rechazan en esos runtimes en
lugar de ignorarse: `tiling`, `overlap_ratio` y `output_file_format` salen con
`config_unsupported` cuando un backend de runtime no puede cumplirlas.

`half` funciona al revés. Los runtimes exportados lo reciben y ejecutan en
FP16; la inferencia nativa en PyTorch registra que se ignoró y continúa en
FP32.

Los modelos de gaze (estimación de la mirada) son de dos etapas y no tienen
detector propio, por lo que `face_detector` es obligatorio para ellos.
`gallery` solo aplica a modelos cuya tarea es `embed`; pasarlo a cualquier otro
sale con `config_unsupported`.

stdout lleva los resultados y nada más; el progreso, las advertencias y los
errores van a stderr. `json=true` imprime un objeto JSON por invocación, o uno
por frame en streaming, cada uno con su `schema_version`. `quiet=true` silencia
stderr. Los dos juntos le dan a un lector automático un stream de stdout
limpio.

El código de salida es `0` en caso de éxito, `2` para un error de uso o de
configuración, `3` cuando la fuente no se encuentra, `4` cuando el modelo no se
puede cargar y `1` para otros fallos en tiempo de ejecución.

`help_json=true` imprime los parámetros, tipos, valores por defecto y flags del
comando como JSON sin ejecutar nada, lo que es la forma fiable de leer esta
tabla desde una versión instalada.

Relacionado: [`libreyolo val`](/docs/cli/val) para métricas medidas sobre un
dataset, [`libreyolo export`](/docs/cli/export) para producir los artefactos de
runtime mencionados arriba.
