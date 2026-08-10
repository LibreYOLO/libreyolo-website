---
title: Rendimiento de inferencia
seo_title: Inferencia más rápida en LibreYOLO
description: >-
  Grafos CUDA, media precisión, batching, inferencia por tiles y aumento en test
  en tiempo de predicción, con los valores por defecto reales y qué familias
  soportan cada uno.
lead: >-
  Cinco controles en tiempo de predicción cambian el throughput o la precisión:
  la reproducción de grafos CUDA, la precisión numérica, el batching, el tiling
  y el aumento en test. Cada uno se aplica a un conjunto concreto de familias, y
  dos de ellos cuestan precisión o latencia en lugar de ahorrarla.
keywords:
  - cuda graphs pytorch inferencia
  - inferencia por batches yolo python
  - inferencia fp16
  - inferencia por tiles objetos pequeños
  - inferencia troceada imágenes grandes
  - tta aumento en test detección
  - capture_graph
  - predecir una carpeta de imágenes yolo
last_verified: 1.5.0
verification: >-
  Valores por defecto de los argumentos de InferenceRunner.__call__ en
  libreyolo/models/base/inference.py. API de grafos CUDA de
  BaseModel.capture_graph, graph_info, release_graphs y cuda_graph_scope en
  libreyolo/models/base/model.py; la adhesión por familia, de la variable de
  clase SUPPORTS_CUDA_GRAPH. Comportamiento de media precisión de
  NOOP_PREDICT_KWARGS en libreyolo/utils/predict_args.py, la advertencia de la
  CLI en libreyolo/cli/commands/predict.py, y CAST_RECIPES más
  SUPPORTED_FAMILIES en libreyolo/quant/api.py. Condiciones de batching de
  InferenceRunner._process_in_batches y _predict_batch. Tiling de _predict_tiled
  y _merge_tile_detections. Aumento en test de BaseModel._predict_augment y
  _merge_tta, con TTA_ENABLED, TTA_SCALES y TTA_FIXED_SIZE leídos en
  libreyolo/models/.
snippets:
  batch:
    - label: Inferencia por batches sobre una carpeta
      language: python
      code: |
        from pathlib import Path
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        folder = Path("batch_demo")
        folder.mkdir(exist_ok=True)
        image = Image.open(SAMPLE_IMAGE)
        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        # Un forward apilado por bloque de 4 en las familias que lo soportan.
        results = model(str(folder), batch=4)
        print(len(results), "results")
    - label: 'Streaming, para que la lista nunca se materialice'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: Capturar por adelantado y luego reproducir (necesita CUDA)
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")


        # Paga el warmup y la captura una sola vez, fuera de la primera
        petición.

        model.capture_graph()


        result = model(SAMPLE_IMAGE, cuda_graph=True)

        print(len(result.boxes))

        print(model.graph_info())
    - label: Capturar solo cuando una forma se repite (necesita CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # "auto" espera a ver una forma dos veces, así que el trabajo de una
        # sola vez nunca paga la captura.
        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())
        model.release_graphs()
  precision:
    - label: Instalar el extra de exportación
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Exportar y volver a cargar, con la precisión por defecto'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: Exportación en FP16 (constrúyela y ejecútala en una máquina con CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: 'FP16 en PyTorch, mediante una receta de cast (necesita CUDA)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Una receta de cast no lee datos de calibración.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Inferencia por tiles sobre una imagen grande
      language: python
      code: >
        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # El tiling solo se activa si la imagen es mayor que el tamaño de
        entrada.

        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))

        large.save("large.jpg")


        model = LibreYOLO("LibreYOLO9s.pt")


        result = model("large.jpg", tiling=True, overlap_ratio=0.2)

        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Aumento en test
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## Los controles y sus valores por defecto

Todos ellos son argumentos de `predict`, y todos vienen desactivados por
defecto.

| Argumento | Por defecto | Efecto |
|---|---|---|
| `batch` | `1` | Imágenes por forward pass, para fuentes de tipo carpeta y lista |
| `cuda_graph` | `False` | Reproduce el forward desde un grafo CUDA capturado |
| `tiling` | `False` | Divide una imagen grande en tiles solapados |
| `overlap_ratio` | `0.2` | Solape entre tiles cuando `tiling` está activo |
| `augment` | `False` | Ejecuta vistas volteadas y las fusiona |
| `half` | | Se acepta, se avisa y se ignora |
| `device` | `None` | Mueve el modelo antes de predecir |

`imgsz` también afecta al coste, porque fija la resolución a la que se ejecuta
el modelo, pero es ante todo un argumento de precisión y va con el modelo más
que aquí.

## Batching

<code-tabs name="batch" />

`batch` se aplica a fuentes de tipo carpeta y lista. Con `batch=1`, cada imagen
ejecuta su propio forward pass. Por encima de `1`, cada bloque se preprocesa, se
apila en un único tensor, se ejecuta de una vez y luego se vuelve a trocear, de
modo que el postproceso de imagen única que ya tiene cada familia recibe lo que
espera.

La vía apilada se toma solo cuando se cumple todo esto:

- `batch` es mayor que `1`
- `tiling` está desactivado
- el aumento en test no está activo
- la familia declara `SUPPORTS_BATCHED_PREDICT`
- la red subyacente no está en modo entrenamiento

La última condición no es un tecnicismo. Una red en modo entrenamiento
normalizaría el bloque apilado con estadísticas de batch cruzadas entre
imágenes, dejando que las imágenes de un mismo bloque se cambiaran unas a otras
las predicciones, así que esas ejecuciones se mantienen secuenciales.

`SUPPORTS_BATCHED_PREDICT` vale true por defecto. Estas familias se descuelgan y
ejecutan una imagen por forward pass sea cual sea el valor de `batch`: Depth
Anything V2, Depth Anything 3, EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net,
LibreMODUS, MiDaS, MoGe-2, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body,
SwinIR, YOLOv1, ZipDepth, todos los detectores de vocabulario abierto y todos
los modelos de visión y lenguaje.

Hay un fallback más. Si el preprocesado no devuelve tensores `(1, C, H, W)`
uniformes, con la misma forma, dtype y dispositivo en todo el bloque, el bloque
se ejecuta de forma secuencial en lugar de apilarse, así que la corrección nunca
depende de que las imágenes resulten tener el mismo tamaño.

Combina `batch` con `stream=True` en una carpeta grande para obtener forward
passes por batch sin mantener todos los resultados en memoria.

## Grafos CUDA

<code-tabs name="graphs" />

Un grafo CUDA graba un forward pass una vez y lo reproduce como un único
lanzamiento. Los detectores pequeños dedican buena parte del tiempo de batch 1 a
lanzar kernels, así que colapsar esos lanzamientos es una ganancia de
throughput, y la salida de la reproducción es idéntica bit a bit a la ejecución
eager.

`cuda_graph` admite tres valores. `False` es el valor por defecto y no hace
nada. `True` captura en el primer uso para cada forma de entrada. `"auto"`
espera a que una forma se repita antes de capturar, así que el trabajo de una
sola vez o con formas cambiantes nunca paga el coste de la captura.

`capture_graph(imgsz=None, batch=1, dtype=None)` saca ese coste de la primera
petición. Un grafo solo es válido para la forma exacta con la que se capturó,
así que aquí `batch` tiene que coincidir con cómo se llame a `predict` después.

`graph_info()` informa de los grafos capturados, del número de reproducciones y
de cualquier motivo por el que la ejecución cayó a eager. `release_graphs()` los
libera junto con sus buffers estáticos.

La captura requiere CUDA y una familia que se haya adherido mediante
`SUPPORTS_CUDA_GRAPH`, porque necesita un forward sin trabajo visible desde el
host y eso se verifica familia por familia. Pedirlo en una familia que no se ha
adherido lanza `NotImplementedError` en lugar de ejecutar en eager en silencio.

Un grafo graba direcciones de memoria, no valores, así que cualquier cosa que
reubique los parámetros lo tira. Cambiar de dispositivo con
`predict(device=...)`, cuantizar y descuantizar invalidan los grafos capturados.

La matriz completa de soporte por familia, las divisiones por costuras y el
contrato de numérica están en [Grafos CUDA](/docs/reference/cuda-graphs).

## Precisión

<code-tabs name="precision" />

`half=True` en tiempo de predicción no hace nada. Se acepta por compatibilidad
con la línea de comandos, lanza una advertencia diciendo que es un no-op y se
descarta antes de llegar a ninguna familia. El flag `--half` de la CLI imprime
la misma advertencia para un modelo `.pt`.

Hay dos vías reales para bajar la precisión.

Para un artefacto exportado, la precisión se elige en el momento de exportar con
`export(format=..., half=True)`, y el archivo resultante se vuelve a cargar con
`LibreYOLO()` sin cambios.

Para la ejecución en PyTorch, `model.quantize(recipe="fp16")` castea el modelo a
float16 e instala hooks que mantienen float32 en las entradas y las salidas del
modelo. `"bf16"` hace lo mismo con bfloat16. Ninguno de los dos casts lee datos
de calibración, así que `calib` se ignora para ellos. La cuantización cubre hoy
cuatro familias: YOLOv9, RF-DETR, BiRefNet y FeyNobg. Un cast en un dispositivo
CPU registra una advertencia de que será lento, así que estas recetas están
pensadas para GPU.

Las dos vías cambian la numérica. Ninguna es garantía de obtener exactamente las
mismas detecciones, así que valida antes de desplegar.

## Inferencia por tiles

<code-tabs name="tiling" />

El tiling recorta una imagen grande en tiles cuadrados solapados, predice sobre
cada uno y fusiona los resultados. Es la opción para objetos pequeños en
imágenes de alta resolución, donde redimensionar la imagen entera encoge los
objetivos por debajo de lo que el modelo puede resolver.

El tamaño de tile es el tamaño de entrada del modelo, o `imgsz` cuando se
indica, y tiene que ser cuadrado. `overlap_ratio` vale `0.2` por defecto. Los
tiles que se solapan se reconcilian con non-maximum suppression por clase al
umbral `iou`, y la lista fusionada se trunca después a `max_det`. Esto significa
que `iou` afecta a las predicciones con tiling incluso en familias que no
ejecutan NMS propia.

El tiling se omite, no es que salga barato, cuando la imagen ya cabe: si ambas
dimensiones están en el tamaño de entrada o por debajo, se ejecuta un forward
pass normal en su lugar. También se omite para clasificación, segmentación
semántica y la tarea `embed`, que caen a una única pasada porque ahí el tiling
no significa nada.

Lanza excepción para las tareas cuyo payload no se puede recomponer: máscaras de
segmentación de instancias, boxes orientados, puntos, profundidad, bordes y
normales. No se puede combinar con `augment`.

El resultado lleva `result.tiled` y `result.num_tiles`. Con `save=True`, las
ejecuciones con tiling escriben un directorio bajo `runs/tiled_detections` con
todos los tiles, la imagen anotada, una visualización en cuadrícula y un
`metadata.json` que registra el tamaño de tile, el solape y los umbrales, con
`result.tiles_path` y `result.grid_path` apuntando a ellos.

## Aumento en test

<code-tabs name="tta" />

`augment=True` ejecuta la imagen más de una vez y fusiona las detecciones con
non-maximum suppression por clase al umbral `iou`. Igual que el tiling, esto
hace que `iou` sea determinante para familias que en otro caso lo ignoran.

En la práctica esto es volteo horizontal. La lista de escalas `TTA_SCALES` vale
por defecto una única escala de `1.0` y ninguna familia incluida la sobrescribe,
así que todas las familias ejecutan dos pasadas: la imagen original y su
reflejo. Las familias marcadas con `TTA_FIXED_SIZE` redimensionan a un cuadrado
fijo, lo que de todos modos convierte el multiescala en un no-op para ellas.

La segmentación semántica y la panóptica hacen otra fusión. Su vista volteada se
vuelve a voltear y las dos distribuciones softmax se promedian antes del argmax,
en lugar de fusionarse como boxes.

El aumento en test no está disponible para todas las tareas. Lanza excepción
para boxes orientados, pose, puntos, profundidad, normales, bordes,
restauración, OCR y modelos de embeddings, y no se puede combinar con el tiling.

Estas familias lo desactivan por completo, así que `augment=True` ejecuta una
única pasada normal: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net,
LibreMODUS, NAFNet, PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2,
SwinIR, TEED, todas las variantes de SAM, todos los detectores de vocabulario
abierto y todos los modelos de visión y lenguaje.

## Medir

Nada en esta página lleva un número de latencia, porque un milisegundo sin su
hardware, su runtime, su precisión y su tamaño de batch no es un dato. Las
cifras medidas en distintos hardware y runtimes se publican en
[visionanalysis.org](https://www.visionanalysis.org), y `libreyolo profile` mide
un modelo concreto en la máquina que tienes delante.
