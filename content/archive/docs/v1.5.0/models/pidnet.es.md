---
title: PIDNet
families:
  - pidnet
seo_title: 'PIDNet: predice y exporta segmentación en tiempo real bajo MIT'
description: >-
  Usa PIDNet en LibreYOLO para segmentación semántica en tiempo real. Instala,
  predice, valida y exporta los checkpoints s/m/l de Cityscapes bajo MIT.
lead: >-
  Una red de segmentación semántica de tres ramas que añade una rama dedicada a
  los bordes sobre un diseño inspirado en el control
  proporcional-integral-derivativo, pensada para inferencia en tiempo real.
  LibreYOLO la incluye solo para segmentación semántica.
keywords:
  - PIDNet
  - segmentación semántica en tiempo real
  - segmentación semántica python
  - Cityscapes
  - predicción densa por píxel
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePIDNets-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) ids de clase

        print(mask.classes)      # ids de clase presentes en la imagen,
        ordenados
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto

        # Results.

        model = LibreYOLO("LibrePIDNets-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Instalación

PIDNet no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local. El sufijo `-sem` en el nombre del archivo es obligatorio para esta
familia.

<code-tabs name="predict" />

La segmentación semántica devuelve un id de clase por píxel, no bounding boxes,
así que `result.semantic_mask` lleva un array `(H, W)` en `.data` y la lista de
ids de clase presentes en la imagen en `.classes`. `conf`, `iou` y `max_det` se
aceptan por paridad de API pero no tienen ningún efecto: el modelo asigna una
clase a cada píxel por argmax, sin umbral de confianza ni paso de NMS. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Tres tamaños, todos con una entrada fija de 1024 px. Los checkpoints publicados
son conversiones de los pesos oficiales de PIDNet para Cityscapes, con 19
clases.

LibreYOLO no entrena PIDNet: `train()` lanza `NotImplementedError` para esta
familia, algo que el [nivel de soporte](/docs/models) de arriba marca como solo
inferencia.

## Validación

`val()` devuelve `metrics/mIoU` y `metrics/pixel_accuracy`, medidos contra
cualquier dataset en el formato con el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. [Exportación](/docs/export) enumera
los argumentos que acepta cada formato.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
