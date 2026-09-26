---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: predice y exporta segmentación semántica con ASPP'
description: >-
  Usa DeepLabv3 en LibreYOLO para segmentación semántica. Instala, predice,
  valida y exporta los checkpoints ResNet y MobileNetV3 de torchvision.
lead: >-
  Una red de segmentación semántica que agrupa características a varias tasas de
  dilatación en paralelo (atrous spatial pyramid pooling) antes de clasificar
  cada píxel. LibreYOLO la incluye solo para segmentación semántica.
keywords:
  - DeepLabv3
  - ASPP
  - atrous spatial pyramid pooling
  - segmentación semántica python
  - segmentar imágenes por píxel
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")

        result = model(SAMPLE_IMAGE, save=True)


        mask = result.semantic_mask

        print(mask.data.shape)   # (H, W) ids de clase

        print(mask.classes)      # ids de clase presentes en la imagen,
        ordenados
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto

        # Results.

        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## Instalación

DeepLabv3 no necesita ningún extra opcional. Todo lo que importa está en la
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

Tres backbones: ResNet-50 dilatada, ResNet-101 dilatada y MobileNetV3-Large
dilatada. Esto es DeepLabv3, no DeepLabv3+, así que no hay etapa de decoder ni
refinamiento con CRF, siguiendo la implementación de torchvision en lugar del
código de referencia del propio paper.

LibreYOLO no entrena DeepLabv3: `train()` lanza `NotImplementedError` para esta
familia, algo que el [nivel de soporte](/docs/models) de arriba marca como solo
inferencia. Los tres checkpoints publicados son los propios pesos de torchvision
entrenados en COCO con las etiquetas de VOC, convertidos para el cargador de
LibreYOLO.

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
