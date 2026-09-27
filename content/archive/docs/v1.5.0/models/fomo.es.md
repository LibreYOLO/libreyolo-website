---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: localización de puntos, entrenamiento y exportación en LibreYOLO'
description: >-
  Ejecuta FOMO (Faster Objects, More Objects) en LibreYOLO: un detector diminuto
  de localización de puntos para contar muchos objetos pequeños. Instala,
  predice, entrena y exporta.
lead: >-
  FOMO es un localizador de puntos basado en una rejilla: cada celda de una
  rejilla de baja resolución se clasifica como fondo o como centro de un objeto,
  sin ninguna regresión de bounding box. LibreYOLO lo incluye para la tarea de
  puntos.
keywords:
  - FOMO
  - Faster Objects More Objects
  - localización de puntos
  - detección de centroides
  - detectar objetos pequeños
  - edge AI
  - detección de objetos en microcontrolador
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Los pesos de LibreFOMO no se descargan solos (ver Checkpoints más
        abajo).

        # Apunta esto a un checkpoint que ya hayas descargado en local.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # Hay que pasar imgsz: la CLI usa 640 por defecto, y el checkpoint s

        # solo acepta su valor nativo de 96.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Instalación

FOMO no necesita nada más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

A diferencia de todas las demás familias de este sitio, los pesos de LibreFOMO
no se descargan automáticamente: `LibreYOLO("LibreFOMOs-point.pt")` busca ese
archivo en disco y lanza un `ValueError` que lo nombra en lugar de traerlo de
Hugging Face. Descarga primero un checkpoint desde la [organización LibreYOLO](https://huggingface.co/LibreYOLO)
y cárgalo por su ruta local, o entrena el tuyo (ver Entrenamiento más abajo).

<code-tabs name="predict" />

El resultado lleva una carga de `points` en lugar de `boxes`: cada fila es
`x, y, clase, confianza`, disponible como `result.points.data` o a través de
los accesores `.xy`, `.xyn`, `.cls` y `.conf`. No hay ningún umbral `iou` que
fijar, porque no hay cajas que suprimir; `predict(..., nms_radius=1)` controla
cuántas celdas de la rejilla deben separar a dos detecciones para que ambas
sobrevivan, y el nombre del archivo tiene que llevar el sufijo de tarea
`-point` de FOMO para que el cargador lo reconozca. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Tres tamaños, `s`, `m` y `l`, usan backbones de estilo MobileNetV2
progresivamente más anchos con resoluciones de entrada fijas correspondientemente
mayores, cada uno detrás de una única cabeza de clasificación 1x1. Esta familia
no lleva aquí una tabla de benchmarks; el tamaño del archivo de checkpoint en la
tabla de más abajo es la señal por tamaño más clara publicada hasta ahora.

## Entrenamiento

<code-tabs name="train" />

`imgsz` no es una elección libre: toma por defecto la resolución nativa del
checkpoint cargado, y pasar un valor distinto lanza un `ValueError` que nombra
el tamaño que espera. Esos tamaños son 96 para `s`, 192 para `m` y 224 para `l`.
La CLI usa 640 por defecto para `imgsz`, así que un comando `libreyolo train`
tiene que fijarlo explícitamente para que coincida con el checkpoint.

Si no lo tocas, el entrenador ejecuta 40 epochs con batch 32 usando Adam a
`lr0=3e-4`, sin weight decay, y con la clase de primer plano ponderada 100x
sobre el fondo en la loss de entropía cruzada por celda, ya que casi todas las
celdas de la rejilla son fondo en una escena típica. EMA y precisión mixta están
desactivadas por defecto, y no se aplica ninguno de los aumentos geométricos o
de color que se usan en el resto de LibreYOLO: mosaic, mixup, jitter de HSV,
volteo, rotación, traslación y shear están todos a cero.

Esta es la ruta con la que se entrenaron los checkpoints publicados de
LibreFOMO, desde cero sobre COCO.

Consulta [entrenamiento](/docs/train) para datasets y loggers.

## Validación

`val()` deriva a un validador a nivel de rejilla hecho para esta familia. Junto
a las claves `metrics/precision`, `metrics/recall` y `metrics/mAP@` de
emparejamiento de puntos que comparte con otras tareas de puntos, barre umbrales
de confianza y valores de `nms_radius` y publica la combinación de mejor F1 bajo
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall` y
`metrics/grid_mean_distance`, además del umbral y el radio que la produjeron
bajo `decode/threshold` y `decode/nms_radius`.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según la extensión
de su archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. También se admite ejecutar el grafo en
un runtime pelado, sin LibreYOLO instalado, pero entonces el preprocesado y el
postprocesado corren de tu cuenta.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia. Ninguno se descarga
automáticamente: consigue el archivo que quieras desde la página de Hugging Face
enlazada y pasa su ruta local a `LibreYOLO()`.

<checkpoint-table />

## Licencia

<provenance-box>

No hay ningún repositorio de código upstream de FOMO al que enlazar: Edge
Impulse describe la técnica en una entrada de blog y en la documentación de su
producto, pero no ha publicado código de entrenamiento ni de inferencia de FOMO.
La arquitectura y el entrenamiento que hay aquí son la implementación propia de
LibreYOLO de esa descripción publicada, y los checkpoints de LibreFOMO
publicados están entrenados desde cero sobre COCO, así que tanto el código como
estos pesos son MIT, propios de LibreYOLO. El nombre FOMO y la técnica que
describe siguen siendo de Edge Impulse.

</provenance-box>
