---
title: API del modelo
seo_title: "Métodos y firmas del objeto modelo de LibreYOLO"
description: "Todos los métodos de un modelo LibreYOLO cargado: predict, embed, track, val, train, export, save, quantize, info y los controles de grafos CUDA, con sus valores por defecto reales."
lead: "Un modelo LibreYOLO cargado es una instancia de BaseModel. Esta página lista los métodos que lleva esa instancia, con las firmas y los valores por defecto leídos de libreyolo/models/base/model.py."
keywords:
  - métodos del modelo libreyolo
  - argumentos de predict libreyolo
  - argumentos de val libreyolo
  - exportar modelo libreyolo
  - model.track
  - model.quantize
  - capture_graph
last_verified: "1.5.0"
verification: "Firmas y valores por defecto leídos de libreyolo/models/base/model.py y libreyolo/models/base/inference.py en la v1.5.0. Las clases de familia pueden restringirlos o ampliarlos; train() se define por familia y aquí solo se documenta su envoltorio común cfg=."
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True devuelve un generador, un Results por frame o imagen.
        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
---

## Construcción

La factoría devuelve una instancia de la clase de familia. Construir esa clase
directamente admite los mismos argumentos, salvo que `size` es obligatorio:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` selecciona CUDA cuando está disponible, luego MPS y luego CPU.
Un entero o una cadena de dígitos se interpreta como un ordinal de CUDA, así que
`device=0` y `device="0"` significan ambos `cuda:0`. `task` se valida contra el
`SUPPORTED_TASKS` de la familia. Pasar `model_path=None` construye la
arquitectura y la deja en modo de entrenamiento; pasar un `dict` carga ese state
dict directamente.

## predict y \_\_call\_\_

`predict` es un alias de `__call__`.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| Argumento | Valor por defecto | Significado |
|---|---|---|
| `source` | `None` | Imagen, lista o tupla de imágenes en memoria, directorio, archivo de vídeo, o una fuente de pantalla como `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Umbral de confianza |
| `iou` | `0.45` | Umbral de IoU para NMS |
| `imgsz` | `None` | Sobrescribe el tamaño de entrada; `None` usa el tamaño nativo del modelo |
| `device` | `None` | Sobrescribe el dispositivo para esta llamada |
| `classes` | `None` | Conserva solo estos IDs de clase |
| `max_det` | `300` | Detecciones máximas por imagen |
| `augment` | `False` | Aumento de datos en tiempo de test |
| `save` | `False` | Escribe una imagen o un vídeo anotados |
| `batch` | `1` | Imágenes por pasada hacia delante para fuentes de tipo directorio y lista |
| `stream` | `False` | Devuelve un generador en lugar de una lista materializada |
| `stream_buffer` | `False` | Conserva todos los frames capturados en vivo en lugar de solo el más reciente |
| `vid_stride` | `1` | Procesa uno de cada N frames de vídeo o de pantalla |
| `show` | `False` | Muestra los frames anotados en una ventana |
| `output_path` | `None` | Ruta de salida cuando `save=True` |
| `color_format` | `"auto"` | Pista de formato de color para arrays en memoria |
| `tiling` | `False` | Inferencia por tiles para imágenes grandes |
| `overlap_ratio` | `0.2` | Ratio de solape entre tiles |
| `output_file_format` | `None` | `"jpg"`, `"png"` o `"webp"` |
| `cuda_graph` | `False` | `True` captura en el primer uso por cada forma de entrada, `"auto"` espera a que una forma se repita |

Una fuente de una sola imagen devuelve un `Results`. Una lista, una tupla o un
directorio devuelven una lista de ellos, y `stream=True` devuelve un generador
en todos los casos.

Las fuentes de streaming en vivo no tienen fin y requieren `stream=True`.
`tiling` y `augment` no se pueden combinar. El aumento de datos en tiempo de
test lanza un error para las tareas `embed`, `point` y `edge`.

<code-tabs name="usage" />

Con `batch > 1`, las familias cuyo `SUPPORTS_BATCHED_PREDICT` es verdadero
ejecutan una única pasada hacia delante apilada por bloque; `batch=1` mantiene
una pasada por imagen.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Un envoltorio de conveniencia sobre `predict` que apila cada fila de embedding
en un único tensor `(N_total, D)`. El modelo tiene que haberse construido con
`task="embed"`; en caso contrario lanza `NotImplementedError`.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Emite un `Results` por frame con `track_id` asignado. `tracker` es
`"bytetrack"`, `"botsort"`, `"ocsort"` o `"deepocsort"`, y se ignora cuando se
pasa `tracker_config`, porque el tipo de configuración selecciona el tracker.
`track_conf` se corresponde con `track_high_thresh` en ByteTrack y BoT-SORT y
con `det_thresh` en OC-SORT y Deep OC-SORT. `output_path` es por defecto
`runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Devuelve un diccionario de métricas cuyas claves dependen de la tarea; la
detección devuelve `metrics/precision`, `metrics/recall`, `metrics/mAP50` y
`metrics/mAP50-95`. `imgsz` acepta un entero cuadrado o una tupla
`(height, width)` y por defecto usa el tamaño de entrada nativo del modelo.
`plots` es un alias de `save_plots`. `allow_download_scripts` controla el Python
embebido que un YAML de dataset puede llevar en su campo `download`.

`faster_coco_eval` se acepta a través de `**kwargs` y por defecto vale `True`,
con vuelta a pycocotools cuando el paquete no está instalado. El backend que se
ejecutó se indica en `model.last_eval_backend`.

La validación con aumento de datos lanza un error para las tareas `obb` y
`pose`.

## train

`train` se define por familia, así que sus argumentos varían. Dos
comportamientos son comunes, porque la clase base envuelve el `train` de cada
familia:

- `cfg=` toma la ruta de un YAML cuyas claves se fusionan en la llamada. Los
  argumentos con nombre explícitos ganan al archivo.
- `pretrained=False` en una familia del grupo de cobertura `g0` o `g1`
  reinicializa el modelo desde cero antes de entrenar, y no se puede combinar
  con `resume=True`.

Qué parámetros de aumento de datos respeta realmente cada familia es una
cuestión de cada una; consulta la
[matriz de aumento de datos](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Devuelve la ruta del artefacto escrito. `format` se resuelve a través del
registro de exportadores, donde `engine` es un alias de `tensorrt` y `litert` es
un alias de `tflite`. Argumentos comunes a todos los exportadores:

| Argumento | Valor por defecto | Significado |
|---|---|---|
| `output_path` | `None` | Ruta del archivo de salida; se genera dentro de `weights/` cuando se omite |
| `imgsz` | `None` | Tupla `(height, width)` o un único entero; por defecto, el tamaño nativo |
| `opset` | `None` | Versión del opset de ONNX |
| `simplify` | `True` | Ejecuta la simplificación del grafo ONNX |
| `dynamic` | `True` | Activa los ejes dinámicos |
| `half` | `False` | Precisión FP16 |
| `int8` | `False` | Precisión INT8 |
| `batch` | `1` | Tamaño de batch fijado en el artefacto |
| `device` | `None` | Dispositivo sobre el que trazar |
| `data` | `None` | data.yaml para la calibración INT8 |
| `fraction` | `1.0` | Fracción del dataset de calibración que se usa |
| `allow_download_scripts` | `False` | Permite Python embebido en las descargas del YAML del dataset |
| `verbose` | `False` | Registro detallado del exportador |

Las combinaciones bloqueadas lanzan `NotImplementedError` en la comprobación
previa, antes del trazado. La cobertura y sus reglas están en la página de la
[matriz de exportación](/docs/reference/export-matrix). Cuando hay adaptadores
LoRA activos, se pliegan en pesos densos, y esa fusión ocurre solo después de
todos los rechazos posibles de la petición.

## save

```python
model.save(path) -> str
```

Escribe un checkpoint de LibreYOLO con esquema v1.0: el state dict más los
metadatos descritos en el
[esquema de checkpoint](/docs/reference/checkpoint-schema). Un modelo cuantizado
lleva además su manifiesto `quant`, de modo que `LibreYOLO(path)` restaura la
estructura cuantizada y las escalas.

## quantize, quant_info y dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Cuantiza in situ y devuelve el modelo. `recipe` es uno de los casts `fp16` y
`bf16`, las recetas de Conv y Linear `int8` y `fp8`, o las recetas solo para
Linear `w4a16`, `w4a8`, `nvfp4`, `mxfp4` e `int2`, que soportan las familias
transformer como RF-DETR. `int2` requiere QAT. `calib` toma una ruta a data.yaml
o el nombre de un dataset integrado y lee las imágenes solo hacia delante; las
etiquetas nunca se leen. Pasa `calib=None` para saltarte la calibración.
`algorithm` es `"minmax"`, `"percentile"` o `"auto"`.

`model.quant_info()` devuelve el resumen del estado de cuantización, o `None`
para un modelo float. `model.dequantize()` restaura los módulos float in situ
manteniendo los pesos maestros entrenados con cuantización, que es el puente de
QAT a `export(format="onnx", int8=True, data=...)`.

## info y capas

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` devuelve un diccionario compatible con JSON y registra un resumen legible
cuando `verbose` es verdadero. `get_available_layer_names` lista las capas que
puede nombrar una configuración de destilación o de extracción de
características.

## Grafos CUDA

Disponible en las familias cuyo atributo de clase `SUPPORTS_CUDA_GRAPH` es
verdadero. La reproducción es idéntica bit a bit a la ejecución eager.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # gestor de contexto
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Un grafo capturado solo es válido para la forma exacta con la que se capturó,
así que `batch` e `imgsz` deben coincidir con la llamada posterior a `predict`.
`capture_graph` saca el coste de la captura de la primera petición. `mode`
acepta `True` o `"on"` para capturar en el primer uso, `"auto"` para esperar a
que una forma se repita, y `False` para no hacer nada. `capture_graph` lanza
`NotImplementedError` cuando la familia no se ha adherido y
`CudaGraphUnavailable` cuando la captura falla.

## Dispositivo y dtype

Los objetos `Results` llevan `.to()`, `.cpu()`, `.cuda()` y `.numpy()`; consulta
[Tipos de Results](/docs/reference/results-types). El modelo en sí se mueve
pasando `device=` a `predict`, o en el momento de la construcción.
