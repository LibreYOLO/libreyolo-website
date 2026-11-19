---
title: Aumento de datos
seo_title: 'Aumento de datos de entrenamiento en LibreYOLO, con ejemplos'
description: >-
  Qué hace cada parámetro de aumento de datos de LibreYOLO, mostrado sobre una
  imagen real por el propio código de entrenamiento de la biblioteca, con su
  rango, su valor por defecto y las familias que lo leen.
lead: >-
  Los aumentos de datos de entrenamiento son transformaciones aleatorias que se
  aplican a cada muestra de entrenamiento. Se configuran con parámetros de
  TrainConfig, y cada familia de modelos ejecuta su propio pipeline, así que un
  parámetro solo modifica las muestras allí donde ese pipeline lo lee.
keywords:
  - aumento de datos yolo
  - data augmentation yolo
  - mosaic augmentation
  - mixup
  - jitter hsv
  - transformación afín aleatoria
  - random erasing
  - detección de objetos pequeños
  - ejemplos de aumento de datos
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
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
  small:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # YOLOX parte de mosaic_scale=(0.1, 2.0), que puede reducir un

        # objeto a una décima parte de su tamaño. (0.5, 1.5) se queda en la
        mitad.

        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # En la CLI, un par son dos números separados por una coma.
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
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
source_hash: 72d22a38d647de0a
---

## Definir los parámetros

Los parámetros de aumento de datos son argumentos normales de `train()`.

<code-tabs name="train" />

Tres de ellos tienen formas más cortas, que aceptan igual `train()` y la CLI:
`mosaic` corresponde a `mosaic_prob`, `fliplr` a `flip_prob` y, en detección,
`mixup` corresponde a `mixup_prob`. Pasar una forma corta y su campo con valores
distintos lanza un error. En la versión 1.8.0, `perspective` no tiene flag en la
CLI, así que se define desde Python.

Los parámetros de clasificación se validan al construir su pipeline, y la CLI
también comprueba `flip_prob`, `flipud`, `cutmix` y `erasing`. `TrainConfig` no
comprueba el rango de los parámetros de detección, así que `degrees=100` entrena
con rotaciones de hasta 100 grados. La [matriz de aumento de
datos](/docs/reference/augmentation-matrix) da el rango de cada parámetro.

## Tres estados, no dos

Que un parámetro haga algo o no depende de la familia. La biblioteca mantiene
una tabla declarativa con eso, y cada entrada es uno de tres estados.

`used` significa que el parámetro llega al pipeline y modifica las muestras.
`ignored` significa que nunca llega al pipeline, así que definirlo no hace nada.
`gated_by_mosaic` significa que solo se aplica a las muestras que tomaron la
rama de mosaico, así que con `mosaic_prob=0` nunca se activa aunque esté
conectado.

Ese tercer estado es el que sorprende a la gente. En un pipeline de estilo YOLOX,
la transformación afín se ejecuta sobre el lienzo del mosaico y MixUp mezcla una
muestra de mosaico, así que `mosaic_prob=0` desactiva en silencio `degrees`,
`translate`, `shear`, `perspective`, `mosaic_scale`, `mixup_prob` y
`mixup_scale`, todos a la vez. El entrenador registra un aviso específico para
el caso de MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

La CLI también avisa de los parámetros ignorados, y lista solo los que hayas
escrito tú:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Llamado desde Python, `train()` no avisa de los parámetros ignorados. La
[matriz de aumento de datos](/docs/reference/augmentation-matrix) tiene el
estado de cada parámetro en cada familia y el valor del que parte cada familia.
La tabla es la spec que se distribuye en `libreyolo/data/augment/spec.py`, que
los propios tests de la biblioteca contrastan con los pipelines, y se puede leer
directamente:

<code-tabs name="support" />

## Qué hace cada parámetro

Las figuras de abajo las producen las propias transformaciones de entrenamiento
de LibreYOLO 1.8.0, ejecutadas sobre la imagen de ejemplo a través de un dataset
de una sola imagen, así que cada tesela del mosaico y cada imagen compañera de
MixUp es la misma foto. El bounding box es el de la persona que LibreYOLO9t
detecta en esa imagen. Debajo de cada panel aparecen el ajuste y el tamaño del
bounding box más grande, en píxeles a la resolución de entrenamiento de 640
píxeles. Sin aumento de datos, la persona mide 291 × 217 px.

Varios parámetros geométricos son rangos: `degrees=10` significa un ángulo
sorteado en cualquier punto entre -10 y +10 grados. Sus paneles fijan el sorteo
en el valor del pie de figura, así que muestran el efecto más fuerte que permite
un ajuste y no una muestra típica. Las figuras de detección salen del pipeline
de YOLOv9. YOLOv7 entrena con él sin cambios, y RT-DETR y RT-DETRv2 usan su
envoltorio de mosaico con una transformación por muestra propia. YOLOX, RTMDet,
PicoDet y FOMO construyen el mosaico y la transformación de la misma forma en un
envoltorio aparte, cuyo MixUp es distinto.

### mosaic_prob

La probabilidad de que una muestra sea un mosaico: cuatro imágenes colocadas en
un lienzo del doble del tamaño de entrada, que la transformación afín reduce
después al tamaño de entrada. En los pipelines de estilo YOLOX, la
transformación y MixUp solo se ejecutan sobre las muestras de mosaico, así que
ambos se detienen cuando este parámetro vale 0. En RT-DETR, RT-DETRv2, PicoDet y
FOMO vale 0 por defecto.

<aug-figure name="mosaic" />

El segundo panel fija la escala de la transformación en 0.5×, con la que cabe el
lienzo entero. Los dos últimos son sorteos normales con los valores por defecto
de YOLOv9, y el pie de figura indica la escala que sorteó cada uno.

### mosaic_scale

El rango del que se sortea el zoom de la transformación. A 1× un objeto conserva
el tamaño que tiene en inferencia, y a 0.5× tiene la mitad. El valor por defecto
base es `(0.1, 2.0)`, que mantienen YOLOX, YOLOv7, RTMDet, PicoDet, RT-DETR,
RT-DETRv2 y FOMO, así que sus objetos pueden reducirse a una décima parte de su
tamaño. YOLOv9 viene con `(0.5, 1.5)`. YOLO-NAS y PP-YOLOE no tienen mosaico y
reutilizan este rango como escala de su transformación afín por muestra.

<aug-figure name="mosaic_scale" />

Por encima de 1×, la salida muestra solo la parte superior izquierda del lienzo,
así que los objetos quedan cortados en el borde o salen del encuadre.

### degrees

El rango de rotación: cada muestra transformada gira un ángulo sorteado entre
`-degrees` y `+degrees`. La rotación pivota sobre la esquina superior izquierda
del lienzo, así que un ángulo grande saca del encuadre buena parte de la imagen.
YOLOX y YOLOv7 mantienen el valor por defecto base de 10, y YOLOv9 viene con 0.

<aug-figure name="degrees" />

### translate

El rango de desplazamiento, como fracción del lado de entrada: cada muestra
transformada se desplaza hasta `translate` veces 640 px en cada eje con
`imgsz=640`. El valor por defecto base es 0.1. YOLO-NAS y PP-YOLOE vienen con
0.25, y RTMDet, PicoDet y FOMO con 0.

<aug-figure name="translate" />

### shear

El rango de cizalladura en grados, sorteado por separado para cada eje. YOLOX y
YOLOv7 mantienen el valor por defecto base de 2, y todas las demás familias que
lo leen vienen con 0.

<aug-figure name="shear" />

### perspective

Una transformación proyectiva sobre la rotación, la escala, la cizalladura y la
traslación. Cada esquina del lienzo se desplaza hacia dentro una cantidad
aleatoria de hasta `min(perspective * 100, 0.2)` veces la mitad del lado del
lienzo, por eje, así que `perspective=0.001` desplaza una esquina como máximo un
5 por ciento del lado a cualquier resolución, y todo valor a partir de `0.002`
alcanza el mismo máximo. El valor por defecto `0` mantiene la transformación
afín, y un valor negativo lanza `ValueError` cuando se transforma la primera
muestra. Los paneles comparten un único sorteo aleatorio, escalado por el valor.
YOLOv9 y sus variantes E2E y P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet,
PicoDet, RT-DETR y RT-DETRv2 lo aplican.

<aug-figure name="perspective" />

### hsv_prob

La probabilidad de un jitter de color HSV. Solo la probabilidad es un parámetro.
El jitter desplaza el tono hasta 5 en la escala de tono de 0 a 179 de OpenCV, y
la saturación y el valor hasta 30 en la escala de 0 a 255, y cada uno de los
tres desplazamientos se omite con una probabilidad de un medio. Todos los
pipelines de detección que leen `hsv_prob` usan esas cantidades. Los tres
últimos paneles son los más visibles de 119 sorteos.

<aug-figure name="hsv" />

### flip_prob y flipud

Las probabilidades de volteo horizontal y vertical. `flip_prob` parte de 0.5 en
todas las familias excepto FOMO, que viene con 0. `flipud` parte de 0. RTMDet,
PicoDet, RT-DETR, RT-DETRv2, FOMO, PP-YOLOE y las familias de estilo DETR no
tienen volteo vertical y lo ignoran. La receta de PP-YOLOE usa en su lugar una
rotación aleatoria de 90 grados.

<aug-figure name="flip" />

### mixup_prob y mixup_scale

MixUp mezcla una segunda muestra con la actual y conserva los bounding boxes de
ambas. En el pipeline de YOLOv9, la compañera es una muestra completa sin
mosaico, la proporción de mezcla se sortea de forma uniforme entre 0 y 1, y
`mixup_scale` no se lee nunca, aunque la spec de la biblioteca lo lista como
condicionado por el mosaico. En el envoltorio de YOLOX, que usan YOLOX, RTMDet,
PicoDet y FOMO, y en el de YOLO-NAS, la compañera se reescala primero por un
factor sorteado de `mixup_scale` y las dos imágenes se promedian con una
proporción fija de 0.5. El valor por defecto base de `mixup_prob` es 1.0, y
YOLOv9 viene con 0.

<aug-figure name="mixup" />

Los dos últimos paneles mezclan una segunda muestra con un mosaico a 0.5×. Sus
pies de figura indican la proporción de mezcla λ, el peso de la muestra de
mosaico.

### no_aug_epochs

`no_aug_epochs` es `used` en todas partes, pero no significa lo mismo en todas
partes. En los pipelines de mosaico desactiva el mosaico y MixUp durante las
últimas épocas. En los pipelines de estilo DETR detiene los aumentos
fotométricos, de zoom-out y de recorte, y da forma al tramo final del schedule.
En los pipelines de clasificación desactiva el aumento automático, erasing,
MixUp y CutMix, y el recorte y los volteos se mantienen.

## Recetas de estilo DETR

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4 y TinyFormer entrenan con una única
transformación de torchvision, sin mosaico ni transformación afín, y lo mismo
hace EdgeCrafter (EC) en detección. Las tres operaciones fuertes de esa
transformación son constantes de la receta y no parámetros:

- `RandomPhotometricDistort` aplica cada cambio de color (brillo de 0.875 a
  1.125, contraste y saturación de 0.5 a 1.5, tono ±0.05 y una permutación
  aleatoria de canales) con probabilidad 0.5.
- `RandomZoomOut`, con probabilidad 0.5, coloca la imagen sobre un lienzo negro
  de hasta cuatro veces su lado, así que los objetos pueden reducirse a una
  cuarta parte de su tamaño.
- `RandomIoUCrop`, aplicado con probabilidad 0.8, recorta a una región de entre
  0.3 y 1.0 de cada lado, elegida para conservar al menos parte de los bounding
  boxes.

Tras estos pasos se descartan los bounding boxes de 1 px o menos en cualquiera
de sus lados. Las tres operaciones se detienen en la época de parada de la
familia, como muy tarde cuando empiezan las `no_aug_epochs` épocas finales. Los
únicos parámetros que lee este pipeline son `flip_prob` y `no_aug_epochs`.

<aug-figure name="detr_ops" />

Tres muestras del inicio de un entrenamiento de D-FINE, con las tres operaciones
activas:

<aug-figure name="detr_samples" />

RF-DETR tiene un pipeline propio: un volteo horizontal; después, con
probabilidad `crop_resize_prob`, un redimensionado del lado más corto a 400, 500
o 600 px y un recorte cuadrado aleatorio de 384 a 600 px; y, por último, un
redimensionado directo a la resolución de entrenamiento cuadrada, que estira la
imagen en lugar de aplicar letterbox. `crop_resize_prob` vale 0.5 por defecto y
se define desde Python.

<aug-figure name="rfdetr_crop" />

## Objetos pequeños

`erasing` no se aplica a la detección. Random Erasing pertenece al pack de
clasificación y ningún pipeline de detección tiene un paso de erasing, así que
un detector entrena igual con cualquier valor de este parámetro, y no hay nada
que desactivar. Estos son los pasos que encogen o eliminan los objetos
pequeños:

- `mosaic_scale`. En los pipelines de mosaico, el tamaño de un objeto se
  multiplica por la escala sorteada, así que el límite inferior fija el tamaño
  mínimo que alcanza. Con el valor por defecto base `(0.1, 2.0)`, un objeto de
  20 px puede quedar en 2 px. Después, el pipeline de YOLOv9 descarta los
  bounding boxes de 2 px o menos en cualquiera de sus lados, y su transformación
  final, como la de YOLOX, descarta los de 1 px o menos en cualquiera de sus
  lados. Los píxeles del objeto se quedan en la imagen sin etiqueta. Un límite inferior de 0.5, como en el
  `(0.5, 1.5)` de YOLOv9, mantiene todo objeto a la mitad de su tamaño de
  inferencia o más.
- `degrees`, `translate`, `shear` y `perspective` empujan los objetos hacia el
  borde, donde sus bounding boxes se recortan y pasan por el mismo filtro de
  tamaño.
- En las recetas de estilo DETR, `RandomZoomOut` puede reducir los objetos a una
  cuarta parte de su tamaño y `RandomIoUCrop` puede dejarlos fuera del recorte.
  Ninguno de los dos es un parámetro, y ambos se detienen en la época de parada.
- El recorte de RF-DETR puede cortar objetos en su borde. `crop_resize_prob=0`
  lo desactiva.
- La resolución de entrenamiento escala todos los objetos: el letterbox
  redimensiona una imagen con un factor de `imgsz` dividido por su lado más
  largo, así que con `imgsz=1280` cada objeto tiene el doble de ancho y de alto
  que a 640. [YOLOv9](/docs/models/yolov9) también tiene una variante P2 con una
  cabeza de stride 4 para objetos pequeños.

<code-tabs name="small" />

## El pack de clasificación

Seis parámetros actúan solo sobre el pipeline de clasificación. Las familias de
detección ignoran los seis.

<code-tabs name="classify" />

`scale` es el rango de área del recorte aleatorio. `scale=0.5` significa
`(0.5, 1.0)`, y un par explícito fija los dos límites. Un límite inferior
pequeño puede dejar el objeto completamente fuera del recorte, como en el tercer
panel.

<aug-figure name="cls_crop" />

`auto_augment` acepta `"randaugment"`, `"autoaugment"`, `"augmix"` o `None`, y
aplica un sorteo de esa política a cada muestra.

<aug-figure name="auto_augment" />

`erasing` es la probabilidad de Random Erasing, en `[0, 1)`. Se ejecuta después
de la normalización y rellena con ceros un rectángulo que cubre del 2 al 33 por
ciento de la imagen, lo que tras la normalización corresponde al color medio de
ImageNet.

<aug-figure name="erasing" />

`mixup` y `cutmix` son probabilidades por batch que producen etiquetas suaves.
Como mucho se ejecuta una por batch, MixUp primero, así que las dos son aditivas
y deben sumar como mucho 1. En los paneles, A y B son dos recortes de la misma
foto, y cada pie de figura indica la etiqueta suave con la que entrena la imagen
mezclada.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup` y `cutmix` vienen desactivados por defecto,
así que el entrenamiento de clasificación no cambia a menos que lo pidas.
`train()` y la CLI dirigen `mixup` según la tarea: la clasificación usa la
mezcla de batches, mientras que la detección usa `mixup_prob`.

`crop_pct=None` conserva la proporción de redimensionado de evaluación de la
familia. Un valor explícito cambia el recorte de evaluación de la pasada de
validación que se ejecuta durante el entrenamiento, mientras que la exportación
conserva el preprocesamiento nativo de la familia.

La CLI expone `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`,
`fliplr` y `flipud`. En clasificación, `mixup` significa mezcla de batches y
vale 0.0 por defecto. Los volteos horizontales valen 0.5 por defecto y los
verticales 0.0.

`no_aug_epochs` desactiva el aumento automático, erasing, MixUp y CutMix durante
las últimas épocas, y conserva el recorte y los volteos. Las recetas de
clasificación ponen este tramo final a 0 por defecto.

## Parámetros específicos de cada familia

Algunos parámetros viven en la subclase de configuración de una familia en lugar
de en la clase base, así que existen solo para esa familia y no tienen flag en
la CLI.

| Familia | Parámetro | Efecto |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilidad del aumento de instancias copy-paste, solo con `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` reutiliza la misma muestra reflejada, `"mixup"` trae una segunda muestra |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilidad de rotación aleatoria de 90 grados |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Límite de ground truth por imagen en las transformaciones de entrenamiento, 300 por defecto |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste para `task="segment"`, solo en modo `"flip"` |
| RF-DETR | `crop_resize_prob` | Probabilidad de recorte y redimensionado aleatorios en el pipeline nativo, 0.5 por defecto |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Probabilidad de recorte y redimensionado aleatorios, solo con `task="segment"` |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilidades del jitter de la ruta de pose y de la transformación afín que tiene en cuenta los keypoints |

`max_labels` es el que pierde datos. Los bounding boxes que pasan del límite se
descartan con un aviso único en lugar de un error, así que las imágenes densas,
como la fotografía aérea, necesitan subirlo. La memoria de asignación de
etiquetas sigue a la imagen más densa del batch, no al límite.

El mosaico y MixUp están desactivados en el entrenamiento con bounding boxes
orientados independientemente de los parámetros, porque el aumento de datos que
tiene en cuenta las esquinas de los bounding boxes rotados no está implementado.

## Relacionado

- [Matriz de aumento de datos](/docs/reference/augmentation-matrix) para el
  estado de cada familia en cada parámetro, los rangos y los valores por defecto
  de cada familia.
- [Hiperparámetros](/docs/train/hyperparameters) para `no_aug_epochs` como
  argumento del schedule y el resto de `train()`.
- [Datasets](/docs/train/datasets) para los formatos de etiquetas que consumen
  estas transformaciones.
