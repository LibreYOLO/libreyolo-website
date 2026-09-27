---
title: Aumento de datos
seo_title: Aumento de datos de entrenamiento en LibreYOLO
description: >-
  Los parámetros de aumento de datos de TrainConfig, las cuatro formas de
  pipeline que hay detrás y la tabla por familia que dice qué parámetros se
  usan, se condicionan o se ignoran.
lead: >-
  El aumento de datos se configura con parámetros de TrainConfig, pero cada
  familia de modelos ejecuta su propio pipeline de entrenamiento, y un pipeline
  que no tiene rama de mosaico ignora mosaic_prob en lugar de aproximarlo.
keywords:
  - aumento de datos yolo
  - data augmentation yolo
  - mosaic augmentation
  - mixup
  - jitter hsv
  - transformación afín aleatoria
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # La CLI escribe mosaic_prob como mosaic y mixup_prob como mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: Leer la tabla de soporte de una familia
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Solo los ignorados
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Pack de clasificación
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## Definir los parámetros

Los parámetros de aumento de datos son argumentos normales de `train()`.

<code-tabs name="train" />

Dos de ellos tienen una forma más corta en la CLI: `mosaic` corresponde a
`mosaic_prob` y `mixup` corresponde a `mixup_prob`. Todos los demás parámetros se
escriben igual en los dos sitios.

## Tres estados, no dos

Que un parámetro haga algo o no depende de la familia. La biblioteca mantiene
una tabla declarativa con eso, y cada entrada es uno de tres estados.

`used` significa que el parámetro llega al pipeline y modifica las muestras.
`ignored` significa que nunca llega al pipeline, así que definirlo no hace nada.
`gated_by_mosaic` significa que solo se aplica a las muestras que tomaron la
rama de mosaico, así que con `mosaic_prob=0` nunca se activa aunque esté
conectado.

Ese tercer estado es el que sorprende a la gente. En un pipeline de estilo YOLOX
la transformación afín se ejecuta sobre el lienzo del mosaico y MixUp mezcla una
muestra de mosaico, así que `mosaic_prob=0` desactiva en silencio `degrees`,
`translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` y
`mixup_scale` de golpe. El
entrenador registra un aviso concretamente para el caso de MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

La CLI también avisa de los parámetros ignorados, y lista solo los que hayas
escrito tú:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Cuatro formas de pipeline

Las familias se agrupan en cuatro pipelines de entrenamiento, y el pipeline
determina casi todas las respuestas.

El pipeline de mosaico de estilo YOLOX aplica el jitter HSV y los volteos por
muestra, y después ejecuta la transformación afín y MixUp dentro de la rama de
mosaico. Cubre YOLOX, YOLOv7, YOLOv9 y sus variantes E2E y P2, RTMDet, PicoDet,
RT-DETR, RT-DETRv2 y FOMO.

El pipeline de paso directo de estilo DETR no tiene mosaico ni transformación
afín. Su distorsión fotométrica, el zoom-out y el recorte por IoU son constantes
de la receta y no parámetros de configuración, así que solo `flip_prob` y
`no_aug_epochs` están activos. Cubre D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4,
EC y, con un cambio, RF-DETR.

El pipeline de clasificación con ImageFolder ignora todos los parámetros de
detección. Su volteo horizontal es un 0.5 fijo al que `flip_prob` no llega. En su
lugar tiene su propio pack de parámetros, descrito más abajo.

YOLO-NAS es una forma en sí misma: nada de mosaico, una transformación afín por
muestra siempre activa y MixUp aplicado de forma independiente en lugar de
condicionado. Su valor de `mosaic_scale` se reutiliza como rango de escala de esa
transformación afín.

SegFormer y NAFNet ejecutan cada uno un pipeline específico de su tarea cuya
aleatoriedad está fijada en la familia en lugar de ser configurable. En SegFormer
los parámetros activos son los atributos de clase `semantic_scale_jitter` y
`semantic_hsv_prob`, no `mosaic_scale` ni `hsv_prob`. El recorte y los volteos de
NAFNet son operaciones acopladas de entrada y objetivo con una probabilidad fija
de 0.5.

## Qué familia respeta qué parámetro

La tabla de abajo es la spec que se distribuye en
`libreyolo/data/augment/spec.py`, que los propios tests de la biblioteca
contrastan con el cableado real del pipeline. Léela ahí en lugar de deducirla de
la arquitectura.

<code-tabs name="support" />

Resumida por pipeline, para los parámetros base:

| Parámetro | Estilo YOLOX | YOLO-NAS | Estilo DETR | Clasificación |
|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored |
| `mixup_prob` | condicionado por el mosaico | used | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored |
| `flip_prob` | used | used | used | ignored |
| `flipud` | used | used | ignored | ignored |
| `degrees` | condicionado por el mosaico | used | ignored | ignored |
| `translate` | condicionado por el mosaico | used | ignored | ignored |
| `shear` | condicionado por el mosaico | used | ignored | ignored |
| `perspective` | condicionado por el mosaico | used | ignored | ignored |
| `mosaic_scale` | condicionado por el mosaico | used | ignored | ignored |
| `mixup_scale` | condicionado por el mosaico | used | ignored | ignored |
| `no_aug_epochs` | used | used | used | used |

Excepciones dentro de esas columnas, todas ellas restrictivas:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 y FOMO no tienen volteo vertical, así que
  `flipud` se ignora. El envoltorio de mosaico de FOMO también se construye sin
  perspectiva.
- El pipeline nativo de RF-DETR no tiene jitter HSV, así que `hsv_prob` se ignora
  además de lo que dice la columna de estilo DETR.
- EC respeta `hsv_prob`, `degrees` y `translate`, pero solo para `task="pose"`,
  cuya transformación, que tiene en cuenta los keypoints, los lee. Sus
  rutas de detect y segment usan recetas fotométricas fijas.
- DINOv2 sigue la columna de estilo DETR para sus tareas detect y semantic, y
  añade el pack de clasificación para `task="classify"`.

`no_aug_epochs` está en `used` en todas partes, pero no significa lo mismo en
todas. En los pipelines de mosaico apaga el mosaico y MixUp durante las épocas
finales. En los pipelines de estilo DETR detiene los aumentos fotométricos, de
zoom-out y de recorte, y da forma a la cola del schedule. En los pipelines de
clasificación y de semántica solo da forma a la cola.

## El pack de clasificación

Cuatro parámetros gobiernan el pipeline de clasificación y nada más. Las familias
de detección ignoran los cuatro.

<code-tabs name="classify" />

`auto_augment` acepta `"randaugment"`, `"autoaugment"`, `"augmix"` o `None`.
`erasing` es la probabilidad de RandomErasing. `mixup` y `cutmix` son
probabilidades por batch que producen etiquetas suaves; como mucho se ejecuta una
por batch, MixUp primero, así que las dos son aditivas y su suma no debería pasar
de 1.

Los cuatro vienen desactivados por defecto, así que el entrenamiento de
clasificación no cambia a menos que lo pidas.

Hay una colisión de nombres que conviene decir con claridad: en la CLI, `mixup`
es el alias del `mixup_prob` de detección. El campo `mixup` de clasificación no
tiene forma propia en la CLI y solo se alcanza a través de
`model.train(mixup=...)` en Python.

## Parámetros específicos de cada familia

Algunos parámetros viven en la subclase de configuración de una familia en lugar
de en la clase base, así que existen solo para esa familia y no tienen flag en la
CLI.

| Familia | Parámetro | Efecto |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilidad del aumento de instancias copy-paste, solo para `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` reutiliza la misma muestra reflejada, `"mixup"` trae una segunda muestra |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilidad de rotación aleatoria de 90 grados |
| YOLOv9 | `max_labels` | Límite de ground truth por imagen en las transformaciones de entrenamiento, 100 por defecto |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste para `task="segment"`, solo en modo `"flip"` |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Probabilidad de recorte y redimensionado aleatorios |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilidades del jitter de la ruta de pose y de la transformación afín que tiene en cuenta los keypoints |

`max_labels` es el que pierde datos en silencio. Los bounding boxes que pasan del
límite se descartan sin error, así que las imágenes densas como la fotografía
aérea necesitan subirlo.

El mosaico y MixUp están desactivados en el entrenamiento con bounding boxes
orientados independientemente de los parámetros, porque el aumento de datos que
tiene en cuenta las esquinas de los bounding boxes rotados no está implementado.

## Relacionado

- [Hiperparámetros](/docs/train/hyperparameters) para `no_aug_epochs` como
  argumento del schedule y el resto de `train()`.
- [Datasets](/docs/train/datasets) para los formatos de etiquetas que consumen estas transformaciones.
