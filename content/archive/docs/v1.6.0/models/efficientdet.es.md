---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: detección de objetos en LibreYOLO'
description: >-
  Ejecuta EfficientDet D0-D4 en LibreYOLO: detectores con BiFPN para predicción,
  validación y exportación a ONNX, TensorRT y OpenVINO bajo licencia Apache-2.0.
lead: >-
  EfficientDet combina un backbone EfficientNet con una red piramidal de
  características bidireccional y repetida (BiFPN), y escala a la vez
  profundidad, anchura y resolución en cinco tamaños. LibreYOLO lo incluye como
  detector solo de inferencia.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - detección de objetos python
  - escalado compuesto
  - exportar efficientdet onnx
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO


        # La factoría enruta según la extensión del archivo, así que un
        artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("LibreEfficientDetd0.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Instalación

EfficientDet no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. EfficientDet decodifica
candidatos basados en anchors y después aplica non-maximum suppression por
clase, así que aquí `conf`, `iou` y `max_det` tienen todos un efecto real.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Cinco tamaños, de D0 a D4. Cada escalón combina un backbone EfficientNet mayor
con una BiFPN más profunda y más ancha y una cabeza de predicción más profunda,
así que el número de parámetros y el cómputo crecen juntos, siguiendo la regla
de escalado compuesto del paper.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según la extensión
del archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

Los checkpoints D0-D4 de LibreYOLO se convierten a través del proyecto
rwightman/efficientdet-pytorch, con licencia Apache-2.0, que a su vez replica
los pesos oficiales entrenados con TensorFlow de google/automl sin modificar los
tensores aprendidos. No se ha consultado ni utilizado código del proyecto
zylo117/Yet-Another-EfficientDet-Pytorch, con licencia LGPL.

</provenance-box>
