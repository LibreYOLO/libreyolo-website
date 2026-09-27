---
title: Detección de puntos
seo_title: Detección de puntos y conteo en LibreYOLO
description: >-
  Localiza objetos como puntos individuales en lugar de cajas en LibreYOLO.
  Predice centroides, cuenta objetos, entrena FOMO y lee las métricas de puntos.
lead: >-
  La detección de puntos devuelve una posición x, y por objeto en lugar de un
  bounding box. LibreYOLO la expone como la tarea point, y una predicción lleva
  una fila de x, y, clase y confianza por objeto.
keywords:
  - detección de puntos python
  - contar objetos python
  - detección de centroides
  - FOMO localización de puntos
  - contar objetos en imágenes
  - conteo de objetos en visión artificial
last_verified: 1.5.0
snippets:
  predict:
    - label: Predecir puntos y contarlos
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Los pesos de LibreFOMO no se descargan automáticamente. Consigue
        primero

        # un checkpoint en https://huggingface.co/LibreYOLO y cárgalo por ruta
        local.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # número de objetos

        print(points.xy)       # centros (N, 2) en píxeles de la imagen original

        print(points.cls, points.conf)
    - label: Coordenadas normalizadas y conteos por clase
      language: python
      code: >
        from collections import Counter


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE)


        points = result.points.numpy()

        print(points.xyn)                          # los mismos centros en [0,
        1]

        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Entrenar FOMO con un dataset YOLO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Predecir con el checkpoint entrenado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("./LibreFOMOs-point.pt")
        results = model.train(data="my-dataset.yaml", epochs=40)

        # train() recarga el mejor checkpoint en el mismo objeto, así que el
        # modelo predice con los pesos entrenados cuando la llamada termina.
        print(results["best_checkpoint"])
        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Validar y leer las claves de las métricas
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness

        print(metrics["metrics/MLE"])               # error medio de
        localización

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # error de
        conteo
    - label: Cambiar los umbrales de distancia
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Los límites del barrido forman parte del texto de la clave, así que un

        # barrido personalizado renombra las claves mAP que produce.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Exportar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Ejecutar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría decide por el sufijo del archivo, así que un artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Definición

La tarea `point` localiza cada objeto con una sola coordenada x, y y una clase,
sin anchura, altura ni máscara. Como una predicción es una lista plana de
objetos, el número de filas es el número de objetos, y eso es lo que convierte a
esta en la tarea de conteo.

Una predicción rellena `result.points`, un payload `Points` que envuelve un
array `(N, 4)` de filas `x, y, clase, confianza` en píxeles de la imagen
original. `.xy` devuelve las coordenadas, `.xyn` esas mismas coordenadas
divididas por el tamaño de la imagen, `.cls` los índices de clase y `.conf` las
puntuaciones; `len()` devuelve el número de puntos. `result.boxes` se queda
vacío, así que `iou` y `max_det` no tienen sobre qué actuar.

## Modelos

Tres familias cubren `point`, y no son intercambiables.

[FOMO](/docs/models/fomo) es la opción de vocabulario fijo: un clasificador de
rejilla que etiqueta cada celda de una rejilla de baja resolución como fondo o
como centro de un objeto. Es la única familia de puntos que LibreYOLO puede
entrenar, y la única que exporta.

[LocateAnything](/docs/models/locate-anything) recibe texto en lugar de un
índice de clase, así que el vocabulario es la frase que tú escribas. Necesita el
extra `vlm`, se construye como `LibreLocateAnything` y no a través de la
factoría `LibreYOLO()`, y sus pesos están restringidos a uso no comercial. Los
términos exactos, y las dos licencias adicionales que el checkpoint compone,
están en su página.

[SenseNova-Vision](/docs/models/sensenova-vision) llega a `point` mediante el
mismo checkpoint de generación con prompts que usa para otras seis tareas,
cargado con `LibreVLM("sensenova-vision", task="point")`. Necesita el extra
`sensenova`, y cada predicción es una pasada de generación sobre un modelo de
7B, así que cuenta con una latencia por imagen bastante más alta que la de un
detector específico. Sus pesos son no comerciales; la licencia está en su
página.

## Predicción

Los pesos de LibreFOMO son la única excepción a la descarga automática en este
sitio. `LibreYOLO("LibreFOMOs-point.pt")` busca ese archivo en disco y lanza un
`ValueError` que lo nombra en lugar de descargarlo. Descarga primero un
checkpoint de la [organización LibreYOLO](https://huggingface.co/LibreYOLO) en
Hugging Face y cárgalo por ruta local, o entrena el tuyo.

<code-tabs name="predict" />

El nombre de archivo tiene que llevar el sufijo de tarea `-point` para que el
cargador lo reconozca. `predict(..., nms_radius=1)` controla cuántas celdas de
rejilla de separación deben tener dos detecciones de FOMO para que sobrevivan
las dos. Consulta [predicción](/docs/predict) para fuentes, streaming y gestión
de resultados.

## Formato del dataset

`point` no tiene un formato de etiquetas propio. Las familias de puntos leen el
diseño estándar de detección de YOLO y derivan un centro de cada fila de caja,
así que `cx cy` es el punto y `w h` solo deciden si la fila es válida.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Cada archivo de etiquetas contiene una fila por objeto, con coordenadas
normalizadas:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Un archivo de etiquetas ausente o vacío significa que no hay objetos. Consulta
[formatos de dataset](/docs/reference/dataset-formats) para el contrato completo.

## Entrenamiento

FOMO es la única familia de puntos con una implementación de entrenamiento.
`train()` en LocateAnything y en SenseNova-Vision lanza `NotImplementedError`;
haz fine-tuning de esos modelos upstream y carga el resultado.

<code-tabs name="train" />

`imgsz` no es una elección libre en FOMO: por defecto toma la resolución nativa
del checkpoint cargado, y pasar un valor distinto lanza un `ValueError` que
nombra el tamaño que espera. Consulta [entrenamiento](/docs/train) para
datasets, loggers y multi-GPU, y la [página de FOMO](/docs/models/fomo) para los
valores por defecto de esta familia.

## Validación

`val()` empareja los puntos predichos con los puntos del ground truth uno a uno
con el algoritmo húngaro, sobre un barrido de umbrales de distancia. Un umbral
es una distancia euclídea en coordenadas de imagen normalizadas, y el barrido
por defecto son diez valores de 0.01 a 0.10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` y `metrics/f1` se promedian por macro
sobre las clases en el umbral más estricto del barrido, 0.01 por defecto.
`metrics/mAP@0.01` es la precisión media en ese mismo umbral, y
`metrics/mAP@[0.01:0.10]` es la media sobre todo el barrido. Ese valor del
barrido es también `fitness`, el número que lee la selección del mejor
checkpoint. Ambas claves de mAP se construyen a partir de los umbrales en uso,
así que pasar `dist_thresholds=` las renombra.

`metrics/MLE` es la distancia media entre pares emparejados en el umbral más
estricto, en las mismas unidades normalizadas. `metrics/MAE` y `metrics/RMSE`
son métricas de conteo más que de localización: miden la diferencia por imagen
entre el número de puntos predichos y el de puntos del ground truth.

FOMO añade sobre estas un segundo grupo, a nivel de rejilla. Hace un barrido de
la confianza y de `nms_radius` y publica la combinación con mejor F1 como
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`,
`metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP` y
`metrics/grid_FN`, con los ajustes que la produjeron bajo `decode/threshold` y
`decode/nms_radius`.

## Exportación

FOMO exporta a través de la ruta de exportación compartida, y un artefacto
exportado se vuelve a cargar con `LibreYOLO()` según su sufijo de archivo, así
que un archivo `.onnx` o `.engine` se comporta como un checkpoint y devuelve los
mismos `Results`.

<code-tabs name="export" />

La cobertura por formato está en la [página de FOMO](/docs/models/fomo) y en la
[matriz de exportación completa](/docs/reference/export-matrix). LocateAnything
y SenseNova-Vision no exportan: `export()` lanza una excepción en ambos, porque
un modelo generativo no tiene un grafo de detección trazable.
