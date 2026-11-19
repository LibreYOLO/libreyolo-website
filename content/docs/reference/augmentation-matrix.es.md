---
title: Matriz de aumento de datos
seo_title: Qué familia de LibreYOLO respeta qué aumento de datos
description: >-
  Cada parámetro de aumento de datos con su rango y su forma en la CLI, el
  estado de cada parámetro en cada una de las 31 familias entrenables y los
  valores por defecto con los que viene cada familia.
lead: >-
  Definir un parámetro de aumento de datos no garantiza que llegue al pipeline.
  Esta página recoge cómo trata cada familia entrenable cada parámetro de
  TrainConfig, y el valor del que parte cada familia, generados a partir de la
  biblioteca en la versión que documenta.
keywords:
  - aumento de datos libreyolo
  - parámetros de aumento de datos
  - valores por defecto de aumento de datos
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - matriz de soporte de aumento de datos
  - parámetros de TrainConfig
last_verified: 1.8.0
verification: >-
  Generado por scripts/build-augmentation/collect.py a partir de LibreYOLO
  1.8.0: la lista de parámetros, los estados y las notas, de
  libreyolo/data/augment/spec.py, que los tests de la biblioteca anclan a los
  pipelines reales; los valores por defecto, de la subclase de TrainConfig de
  cada familia; y las formas de la CLI, del comando libreyolo train. El mismo
  script contrasta con la biblioteca los rangos y el único punto en el que la
  spec y un pipeline discrepan.
snippets:
  usage:
    - label: Consultar la spec directamente
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
    - label: Los valores por defecto de una familia
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## Los parámetros

Estos son nombres de campos de `TrainConfig`. `train()` y la CLI también aceptan
las formas cortas de la columna CLI, así que `mosaic` define `mosaic_prob`. En
la CLI, `mixup` se dirige según la tarea: mezcla de batches de clasificación
para los clasificadores, `mixup_prob` para los detectores. `perspective` no
tiene flag en la CLI y se define desde Python.

<aug-knobs />

El valor por defecto base es el valor definido en el propio `TrainConfig`. La
mayoría de las familias sobrescriben algunos de ellos, y [Valores por defecto
por familia](#defaults-by-family) da los valores de los que parte realmente un
entrenamiento. Para ver imágenes de lo que hace cada parámetro, consulta
[Aumento de datos](/docs/train/augmentations).

Los rangos entre corchetes son intervalos, así que `[0, 1)` incluye el 0 y
excluye el 1. Los parámetros de clasificación se comprueban al construir su
pipeline y lanzan `ValueError` fuera de su rango. `TrainConfig` no comprueba los
parámetros de detección, así que con un valor fuera de rango se entrena sin
ningún aviso. Las excepciones son una escala sorteada de 0 o menos y un
`perspective` negativo, que lanzan `ValueError` cuando se transforma una
muestra.

<code-tabs name="usage" />

## Los tres estados

| Estado | Significado |
|---|---|
| `used` | El parámetro llega al pipeline de entrenamiento de la familia y modifica las muestras |
| `gated_by_mosaic` | El parámetro solo se aplica a las muestras que tomaron la rama de mosaico, así que con `mosaic_prob == 0` nunca se activa |
| `ignored` | El parámetro nunca llega al pipeline; definirlo no hace nada |

`ignored` es el que conviene comprobar antes de lanzar un entrenamiento, porque
no falla nada. La CLI avisa cuando un parámetro de entrenamiento definido
explícitamente es uno que la familia seleccionada ignora, y el entrenador avisa
cuando `mixup_prob > 0` no puede activarse porque la familia condiciona MixUp al
mosaico y `mosaic_prob` es cero.

## Qué familia respeta qué parámetro

<aug-matrix />

Todas las familias cubiertas siguen uno de seis pipelines. En el pipeline de
estilo YOLOX, el preprocesado por muestra aplica el jitter HSV y los volteos,
mientras que la transformación afín y MixUp solo se ejecutan dentro de la rama
de mosaico. YOLO-NAS, en cambio, ejecuta una transformación afín por muestra que
está siempre activa, ignora el mosaico y aplica MixUp de forma independiente,
reutilizando `mosaic_scale` como rango de escala de la transformación afín.

El pipeline de estilo DETR es una transformación de paso directo sin mosaico. Su
distorsión fotométrica, el zoom-out y el recorte por IoU son constantes de la
receta y no parámetros configurables, y por eso `hsv_prob` y los parámetros
geométricos nunca llegan a él. La clasificación usa `flip_prob` para los volteos
horizontales y `flipud` para los verticales. El jitter de escala y el HSV de las
familias semánticas vienen de atributos de clase de la familia y no de
parámetros de configuración, y los volteos de restauración son operaciones
acopladas sobre entrada y objetivo con una probabilidad fija de 0.5.

`no_aug_epochs` se respeta en todas partes, aunque lo que desactiva varía:
mosaico y MixUp en el estilo YOLOX, la transformación afín y MixUp en YOLO-NAS,
los aumentos fuertes fotométricos y de recorte, más el tramo final del learning
rate, en el estilo DETR, y el aumento automático, erasing, MixUp y CutMix en
clasificación. El recorte y los volteos de clasificación siguen activos.

Donde una familia difiere del resto de su pipeline, el motivo es uno de estos:

| Familia | Diferencia |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | `flipud` ignorado: sus transformaciones no tienen volteo vertical |
| `fomo` | `perspective` y `flipud` ignorados |
| `ppyoloe` | `flipud` ignorado: la receta de origen usa en su lugar una rotación aleatoria de 90 grados |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | `mixup_scale` ignorado: su MixUp mezcla imágenes completas con una proporción aleatoria sin reescalar ninguna de las dos |
| `gtr` | `mosaic_prob` y `mixup_prob` se usan, durante las primeras `mosaic_epochs` épocas; `degrees`, `translate` y `mosaic_scale` dan forma a la transformación afín que sigue al mosaico |
| `rfdetr` | `hsv_prob` ignorado: el pipeline nativo no tiene jitter HSV |
| `ec` | `hsv_prob`, `degrees` y `translate` se usan, solo para `task="pose"`; detect y segment usan recetas fotométricas fijas |
| `dinov2` | El pack de clasificación se usa, solo para `task="classify"` |

`ec` y `dinov2` son familias multitarea, así que un parámetro se marca como
ignorado solo cuando todas las tareas entrenables de la familia lo ignoran. Eso
evita que el aviso de la CLI llegue a ser incorrecto para una tarea y correcto
para otra.

Dome-DETR hereda las transformaciones de D-FINE sin cambios. Lo único que no
admite es el entrenamiento multiescala, que desactiva su propia configuración y
no la spec de aumento de datos.

Hay treinta y una familias cubiertas. Para una familia que no esté en esta
lista, `ignored_aug_params` devuelve un conjunto vacío, así que no se emite
ningún aviso para ella.

## Valores por defecto por familia

El valor del que parte la configuración de cada familia de detección. Una celda
vacía significa que la familia ignora ese parámetro, así que su valor por
defecto no cambia nada.

<aug-defaults />

Dos patrones importan más que cualquier valor concreto. RT-DETR, RT-DETRv2,
PicoDet y FOMO parten de `mosaic_prob=0`, así que en esas familias los
parámetros marcados como condicionados en la matriz no hacen nada hasta que se
activa el mosaico. YOLOv9 parte con la rotación, la cizalladura y MixUp
desactivados y con `mosaic_scale=(0.5, 1.5)`, mientras que el valor por defecto
base es `(0.1, 2.0)`.

EdgeCrafter lee `hsv_prob`, `degrees` y `translate` solo con `task="pose"`, así
que su fila muestra los valores por defecto de la tarea de pose para esos tres
parámetros.

Las familias de clasificación parten de los valores por defecto base:
`flip_prob=0.5`, `flipud=0.0`, `scale=(0.5, 1.0)`, el `crop_pct` propio de la
familia, y `auto_augment`, `erasing`, `mixup` y `cutmix` desactivados.

## Parámetros específicos de cada familia

Algunas familias llevan parámetros de aumento de datos en su propia subclase de
`TrainConfig` en lugar de en la base. La CLI no los expone; defínelos a través
de la API de Python.

| Familia | Parámetro | Significado |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Probabilidad del aumento de instancias copy-paste, solo para `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Origen del copy-paste: `flip` refleja la misma muestra, `mixup` usa una segunda muestra |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Probabilidad de rotación aleatoria de 90 grados |
| `rfdetr` | `copy_paste` | Probabilidad de copy-paste para `task="segment"`, solo en modo `flip` |
| `rfdetr` | `copy_paste_mode` | Modo de origen del copy-paste para `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Probabilidad de recorte y redimensionado aleatorios en el pipeline nativo |
| `dfine` | `crop_resize_prob` | Probabilidad de recorte y redimensionado aleatorios, `task="segment"` |
| `ec` | `crop_resize_prob` | Probabilidad de recorte y redimensionado aleatorios, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Probabilidad de jitter de brillo y contraste, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Probabilidad de la transformación afín que tiene en cuenta los keypoints, `task="pose"` |

`rot90` se aplica a detect y a OBB en `yolo9`.

## Consultar la spec

| Helper | Devuelve |
|---|---|
| `aug_support(family)` | La tabla de parámetro a `Support`, o `None` para una familia desconocida |
| `ignored_aug_params(family)` | El conjunto de nombres de parámetros que la familia ignora; vacío para una familia desconocida |
| `uses_mosaic_gating(family)` | Si el MixUp de la familia solo se activa en muestras de mosaico |
| `display_name(family)` | El nombre de familia de cara al usuario que se usa en los avisos |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | El texto del aviso cuando MixUp no puede activarse nunca, o `None` en caso contrario |

Un `Support` es una tupla con nombre de `status` y `note`, donde la nota explica
por qué un parámetro está ignorado o condicionado para esa familia. En la 1.8.0,
la spec todavía lista `mixup_scale` como condicionado para las familias que
entrenan con el envoltorio de mosaico de YOLOv9, así que `aug_support` lo
devuelve como condicionado en ellas aunque la matriz de arriba, que se contrasta
con el pipeline, lo marca como ignorado.

## La condición del mosaico

En una familia de estilo YOLOX, `mixup_prob=0.5` con `mosaic_prob=0` desactiva
MixUp por completo, porque MixUp solo se aplica a las muestras de mosaico. Es
una combinación fácil de alcanzar al desactivar el mosaico en la parte final del
entrenamiento. El entrenador registra un aviso que nombra a la familia, y
`mixup_gating_warning` es la función pura que hay detrás.
