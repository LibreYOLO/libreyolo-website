---
title: FCN
families:
  - fcn
seo_title: 'FCN: predice y exporta un ResNet FCN bajo BSD-3-Clause'
description: >-
  Usa FCN en LibreYOLO para segmentación semántica. Instala, predice, valida y
  exporta los checkpoints FCN con ResNet dilatada de torchvision.
lead: >-
  Un clasificador denso píxel a píxel que sustituye las capas totalmente
  conectadas de un detector por convoluciones, de modo que produce un mapa de
  clases a resolución completa en lugar de bounding boxes. LibreYOLO lo incluye
  solo para segmentación semántica.
keywords:
  - FCN
  - red totalmente convolucional
  - segmentación semántica python
  - predicción densa por píxel
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreFCNr50.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) ids de clase

        print(mask.classes)      # ids de clase presentes en la imagen,
        ordenados
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto

        # Results.

        model = LibreYOLO("LibreFCNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Instalación

FCN no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

La segmentación semántica devuelve un id de clase por píxel, no bounding boxes,
así que `result.semantic_mask` lleva un array `(H, W)` en `.data` y la lista de
ids de clase presentes en la imagen en `.classes`. `conf`, `iou` y `max_det` se
aceptan por paridad de API pero no tienen ningún efecto: el modelo asigna una
clase a cada píxel por argmax, sin umbral de confianza ni paso de NMS. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Dos profundidades de ResNet, ambas con una entrada fija de 520 px. El grafo de
inferencia de la biblioteca es el FCN con ResNet dilatada de torchvision, no la
red FCN-8s basada en VGG con conexiones skip del paper original.

LibreYOLO no entrena FCN: `train()` lanza `NotImplementedError` para esta
familia, algo que el [nivel de soporte](/docs/models) de arriba marca como solo
inferencia. Los dos checkpoints publicados son los propios pesos de torchvision
entrenados en COCO, convertidos para el cargador de LibreYOLO.

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
