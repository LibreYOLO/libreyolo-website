---
title: Validación y métricas
seo_title: Validación y métricas en LibreYOLO
description: >-
  Ejecuta val() sobre cualquier modelo, consulta las claves de métricas que
  devuelve cada tarea, elige un backend de evaluación y activa una loss de
  validación junto a la métrica de precisión.
lead: >-
  La validación pasa un modelo por un split de un dataset mediante val() y
  devuelve un diccionario plano de claves de métricas y valores float. Las
  claves son cadenas literales, y cuáles obtienes depende de la tarea, no de la
  familia.
keywords:
  - map50-95
  - evaluacion coco
  - metricas de validacion
  - faster-coco-eval
  - pycocotools
  - loss de validacion
  - miou
  - panoptic quality
  - accuracy top1
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: En otro split
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Escribir predicciones en formato COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Ejecutar una validación

`val()` recibe el dataset y devuelve las métricas.

<code-tabs name="val" />

El valor devuelto es un `dict[str, float]` simple. Todas las claves son
literales, así que léelas por nombre y no por posición.

Los argumentos principales son `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` y `verbose`. `conf` vale `0.001` por
defecto e `iou` `0.6`, ambos mucho más permisivos que los valores por defecto de
predicción, porque un barrido de mAP necesita la cola de baja confianza. `imgsz`
toma por defecto el tamaño de entrada del propio modelo en lugar de un número
fijo. `split` acepta `val`, `test` o `train` y nada más.

Cualquier otro campo de la configuración de validación se pasa como argumento
con nombre, incluidos `save_dir`, `max_det`, `eval_max_det`, `half`,
`amp_dtype`, `cache` y `save_plots`.

## Claves de métricas por tarea

La detección devuelve la familia de números de COCO:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Dos de ellas son trampas. `metrics/precision` y `metrics/recall` son alias que se
mantienen por retrocompatibilidad: llevan los valores de mAP 50-95 y
AR@100, no un par de precisión y recall. Usa las claves con nombre.

La segmentación de instancias devuelve las cifras de mAP y AR anteriores como
números de máscara bajo las claves sin sufijo, con las versiones de caja bajo el
sufijo `(B)` y las versiones de máscara repetidas bajo `(M)`. En esta tarea, la
precisión y el recall solo existen con sufijo, como
`metrics/precision(B)`/`metrics/recall(B)` y
`metrics/precision(M)`/`metrics/recall(M)`, y ambos pares llevan los mismos
valores alias que los de detect: el par `(B)` es el mAP50-95 de cajas y el AR@100
de cajas, y el par `(M)` es el mAP50-95 de máscaras y el AR@100 de máscaras.

| Tarea | Claves |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, más los desgloses por tamaño y de recall anteriores |
| segment | versiones de máscara de las claves de detect anteriores (las claves sin sufijo son las de máscara); `precision`/`recall` solo existen como `(B)`/`(M)`, ambos con el mismo alias |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L`, y las claves `keypoints_AR` correspondientes |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, más copias con sufijo `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, más una clave de barrido de mAP |

Los `metrics/precision` y `metrics/recall` de OBB no son alias: son la precisión
y el recall reales con IoU 0.50, tomados en el punto de operación más permisivo
(toda predicción que sobrevive a `conf`, por defecto `0.001`). Las copias con
sufijo `(OBB)` repiten los mismos cuatro valores bajo un nombre específico de la
tarea, la misma convención que `(B)` y `(M)` de arriba.

`accuracy_top5` es en realidad top-`min(5, num_classes)`, así que en un dataset
de tres clases es top-3, algo que cumple toda muestra y que por tanto marca 1.0.

La clave de barrido de la tarea point se construye a partir de los umbrales de
distancia, así que con los valores por defecto queda `metrics/mAP@[0.01:0.10]` y
la clave de umbral único queda `metrics/mAP@0.01`. Pasar `dist_thresholds` cambia
ambas cadenas.

La mayoría de las tareas devuelven además una clave `fitness`, el número único
que usa por defecto la selección del mejor checkpoint. Detección, segmentación y
OBB no la llevan; sus familias se seleccionan por `metrics/mAP50-95`, que sus
diccionarios sí devuelven. Pose no devuelve ni `fitness` ni `metrics/mAP50-95`;
en su lugar, sus entrenadores fijan `best_metric_key` a
`metrics/keypoints_mAP50-95`.

## Claves de velocidad

Todos los validadores añaden tiempos:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Son milisegundos por imagen promediados sobre la ejecución. Describen la máquina
y los ajustes con los que ejecutaste la validación, así que una cifra sacada de
ahí solo tiene sentido si se indica junto con su hardware, su tamaño de batch y
su precisión.

## Backend de evaluación

Las métricas de detección y segmentación se calculan a través de un evaluador
COCO, y `faster_coco_eval=True`, el valor por defecto, selecciona el backend en
C++ cuando el paquete `faster-coco-eval` está instalado. Cuando no lo está, la
ejecución recurre a pycocotools con un aviso por proceso:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Qué backend se ejecutó realmente queda registrado en el modelo como
`last_eval_backend`, y la CLI lo muestra en su salida para las tareas de tipo
detección. Define `LIBREYOLO_FASTER_COCO_EVAL` para sobrescribir el valor de la
configuración desde el entorno.

`iou_thresholds` solo se respeta en la ruta de OBB. La ruta de COCO evalúa con su
propio barrido fijo de 0.50 a 0.95 e ignora el valor.

## Loss de validación

Por defecto la validación informa solo de la precisión. `val_loss=True` calcula
además el objetivo de entrenamiento de la familia sobre los batches de
validación.

<code-tabs name="valloss" />

Emite `metrics/loss` (la loss, es decir, la función de pérdida) más un
`metrics/loss/<component>` por cada término, ponderados exactamente igual que los
pondera el entrenamiento, de modo que los componentes suman el total. A través de
un logger aparecen como `val/loss` y `val/loss/<component>`, y `libreyolo
monitor` superpone `metrics/loss` con `train/loss`.

Los componentes son los propios de cada familia:

| Tarea | Familias | Componentes |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

Está desactivada por defecto porque la asignación de targets añade tiempo y
memoria a la validación. El validador reutiliza la salida del modelo que ya se
produjo para la métrica de precisión en lugar de hacer un segundo forward, se
ejecuta bajo `no_grad` sobre el modelo de evaluación o EMA, y en
entrenamiento multi-GPU se calcula localmente en el rank 0 sin operaciones
colectivas. La selección del mejor checkpoint se mantiene sobre la métrica de
precisión.

Hay tres cosas que deliberadamente no hace. Nunca incluye términos de
contrastive denoising, porque esos necesitan el ground truth en el momento del
forward y la validación hace el forward sin él. Informa del modelo en modo
evaluación, así que donde el forward de entrenamiento y el de evaluación de una
familia difieren de verdad, en las estadísticas de BatchNorm o en la stochastic
depth, el número refleja el modo evaluación; esa es la comparación buscada. Y una
tarea para la que una familia no lo ha implementado lanza un error de
configuración durante la preparación en lugar de saltárselo en silencio:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO es la excepción que no cambia nada: su validador siempre calculó esta loss,
y `val_loss=True` solo afecta a las claves bajo las que se publica.

La validación con aumento de datos y la loss de validación no se pueden
combinar, y pedir ambas lanza un error.

## Archivos que escribe una validación

`val()` siempre escribe `config.yaml` en su directorio de guardado, que por
defecto es `runs/val/<model>_<size>_<timestamp>` cuando no se pasa `save_dir`.

<code-tabs name="json" />

`save_json=True` escribe `predictions.json` para detección, y
`predictions_bbox.json` más `predictions_masks.json` para segmentación. OBB no lo
admite y lo dice.

`save_plots=True` escribe en un subdirectorio `plots/`. La detección obtiene
`box_metrics.png`, gráficas de AP y de recall por clase, curvas de
precisión-recall y de confianza, una matriz de confusión e imágenes de muestra
anotadas cuando OpenCV está instalado. La segmentación añade la copia de máscara
de cada una, y pose tiene su propio conjunto de métricas y curvas.
Los demás validadores no implementan gráficas; classification, semantic,
panoptic, depth, normal, edge, restore, matte, OCR, OBB y point no escriben nada
ahí. Un fallo al generar las gráficas avisa y nunca aborta la ejecución.

## Validación durante el entrenamiento

El entrenamiento valida cada `eval_interval` épocas sobre el split `val` del
dataset, y las métricas que produce son las que gobiernan la selección de
`best.pt`, la parada temprana por `patience` y las claves `val/` de todos los
loggers. La validación se ejecuta sobre los pesos EMA cuando EMA está activado.

Consulta [Hiperparámetros](/docs/train/hyperparameters) para `eval_interval`,
`patience` y `save_plots`, y [Loggers de experimentos](/docs/train/loggers) para
saber dónde van a parar los números.

## Relacionado

- [Datasets](/docs/train/datasets) para las claves de split y los formatos que leen los validadores.
