---
title: Matriz de aumento de datos
seo_title: Qué familia de LibreYOLO respeta cada parámetro de aumento de datos
description: >-
  Soporte de los parámetros de aumento de datos por familia: los dieciséis
  parámetros de TrainConfig, los tres estados, los seis arquetipos de pipeline y
  los parámetros que cada familia ignora en silencio.
lead: >-
  Definir un parámetro de aumento de datos no garantiza que llegue al pipeline.
  Esta página recoge cómo trata cada familia entrenable cada parámetro de
  TrainConfig, a partir de la tabla declarativa que la biblioteca incluye como
  única fuente de verdad.
keywords:
  - aumento de datos libreyolo
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - matriz de soporte de aumento de datos
  - parámetros de TrainConfig
last_verified: 1.5.0
verification: >-
  La lista de parámetros, los estados, los arquetipos, las desviaciones por
  familia y las funciones helper están leídos de libreyolo/data/augment/spec.py
  en la v1.5.0. Esa tabla está anclada a los pipelines reales por
  tests/unit/test_augment_spec.py.
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
source_hash: d2e1b9f5c81072e1
---

## Los parámetros

Estos son nombres de campos de `TrainConfig`, no la forma en que los escribe la
CLI. La CLI mapea sus propios alias sobre ellos, así que `--mosaic` define
`mosaic_prob`.

| Parámetro | Significado |
|---|---|
| `mosaic_prob` | Probabilidad de construir una muestra en mosaico de 4 imágenes |
| `mixup_prob` | Probabilidad de mezclar una segunda muestra |
| `hsv_prob` | Probabilidad de jitter de color HSV |
| `flip_prob` | Probabilidad de volteo horizontal |
| `degrees` | Rango de rotación aleatoria para el warp afín, en grados |
| `translate` | Fracción de traslación aleatoria para el warp afín |
| `mosaic_scale` | Rango de escala aleatoria para el warp afín |
| `mixup_scale` | Rango de escala de jitter aplicado a la imagen compañera de MixUp |
| `shear` | Rango de cizalladura aleatoria para el warp afín, en grados |
| `perspective` | Magnitud del warp proyectivo para el warp afín |
| `flipud` | Probabilidad de volteo vertical |
| `no_aug_epochs` | Épocas finales entrenadas con el aumento de datos fuerte desactivado |
| `auto_augment` | Política AutoAugment de clasificación: randaugment, autoaugment o augmix |
| `erasing` | Probabilidad de RandomErasing en clasificación |
| `mixup` | Probabilidad de MixUp por batch en clasificación, con etiquetas suaves |
| `cutmix` | Probabilidad de CutMix por batch en clasificación, con etiquetas suaves |

Los cuatro últimos son el pack de clasificación. Las familias de detección los
ignoran. `mixup` es un parámetro exclusivo de la API: en la CLI, `--mixup` es el
alias del `mixup_prob` de detección.

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

## Arquetipos de pipeline

Todas las familias cubiertas siguen uno de seis pipelines, con unas pocas
desviaciones por familia que se listan más abajo.

| Parámetro | Estilo YOLOX | YOLO-NAS | Estilo DETR | Clasificación | Semántica | Restauración |
|---|---|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored | ignored | ignored |
| `mixup_prob` | gated | used | ignored | ignored | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored | ignored | ignored |
| `flip_prob` | used | used | used | ignored | ignored | ignored |
| `degrees` | gated | used | ignored | ignored | ignored | ignored |
| `translate` | gated | used | ignored | ignored | ignored | ignored |
| `mosaic_scale` | gated | used | ignored | ignored | ignored | ignored |
| `mixup_scale` | gated | used | ignored | ignored | ignored | ignored |
| `shear` | gated | used | ignored | ignored | ignored | ignored |
| `perspective` | gated | used | ignored | ignored | ignored | ignored |
| `flipud` | used | used | ignored | ignored | ignored | ignored |
| `no_aug_epochs` | used | used | used | used | used | used |
| `auto_augment` | ignored | ignored | ignored | used | ignored | ignored |
| `erasing` | ignored | ignored | ignored | used | ignored | ignored |
| `mixup` | ignored | ignored | ignored | used | ignored | ignored |
| `cutmix` | ignored | ignored | ignored | used | ignored | ignored |

En el pipeline de estilo YOLOX, el preprocesado por muestra aplica el jitter HSV
y los volteos, mientras que el warp afín y MixUp solo se ejecutan dentro de la
rama de mosaico. YOLO-NAS, en cambio, ejecuta un afín por muestra que está
siempre activo, ignora el mosaico y aplica MixUp de forma independiente,
reutilizando `mosaic_scale` como rango de escala del afín.

El pipeline de estilo DETR es una transformación de paso directo, sin mosaico.
Su distorsión fotométrica, el zoom-out y el recorte por IoU son constantes de la
receta y no parámetros configurables, y por eso `hsv_prob` y los parámetros de
geometría nunca llegan hasta él. El pipeline de clasificación usa una
transformación de ImageFolder cuyo volteo horizontal es un 0.5 fijo en lugar de
`flip_prob`. El jitter de escala y el HSV de la semántica vienen de atributos de
clase de la familia y no de parámetros de configuración, y los volteos de
restauración son operaciones acopladas de entrada y objetivo con una
probabilidad fija de 0.5.

`no_aug_epochs` se respeta en todas partes, aunque lo que desactiva cambia: el
mosaico y MixUp en el estilo YOLOX, el afín y MixUp en YOLO-NAS, los aumentos
fotométricos fuertes y los de recorte más la cola del learning rate en el estilo
DETR, y la cola del scheduler en el resto.

## Familias por arquetipo

| Arquetipo | Familias |
|---|---|
| Estilo YOLOX | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| Estilo DETR | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Clasificación | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Semántica | `segformer` |
| Restauración | `nafnet` |

Hay veinticinco familias cubiertas. Una familia que no esté en esta lista
devuelve un conjunto de ignorados vacío, así que no se emite ningún aviso para
ella.

## Desviaciones

| Familia | Diferencia respecto a su arquetipo |
|---|---|
| `rtmdet` | `flipud` ignorado: su transformación no tiene volteo vertical |
| `picodet` | `flipud` ignorado |
| `rtdetr` | `flipud` ignorado |
| `rtdetrv2` | `flipud` ignorado |
| `fomo` | `perspective` y `flipud` ignorados |
| `ec` | `hsv_prob`, `degrees` y `translate` se usan, solo para `task="pose"`; detect y segment usan recetas fotométricas fijas |
| `dinov2` | El pack de clasificación se usa, solo para `task="classify"` |

`ec` y `dinov2` son familias multitarea, así que un parámetro se marca como
ignorado solo cuando todas y cada una de las tareas entrenables de la familia lo
ignoran. Eso evita que el aviso de la CLI llegue a ser incorrecto para una tarea
y correcto para otra.

Dome-DETR hereda las transformaciones de D-FINE sin cambios. Lo único que no
admite es el entrenamiento multiescala, que desactiva su propia configuración y
no la spec de aumento de datos.

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
| `ec`, `yolonas` | `affine_prob` | Probabilidad del afín consciente de keypoints, `task="pose"` |

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
por qué un parámetro está ignorado o condicionado para esa familia.

## La condición del mosaico

En una familia de estilo YOLOX, `mixup_prob=0.5` con `mosaic_prob=0` desactiva
MixUp por completo, porque MixUp solo se aplica a las muestras de mosaico. Es
una combinación fácil de alcanzar cuando se apaga el mosaico en la parte final
del entrenamiento. El entrenador registra un aviso que nombra a la familia, y
`mixup_gating_warning` es la función pura que hay detrás.
