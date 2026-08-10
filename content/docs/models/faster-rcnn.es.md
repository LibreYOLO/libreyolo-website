---
title: Faster R-CNN
families: [faster_rcnn]
seo_title: "Faster R-CNN en LibreYOLO: predecir, validar y exportar"
description: "Ejecuta Faster R-CNN en LibreYOLO para detección de objetos con cuatro backbones. Instala, predice, valida y exporta el port de torchvision con licencia BSD-3-Clause."
lead: "Faster R-CNN detecta objetos con una red de propuestas de regiones que alimenta a un clasificador de dos etapas, la arquitectura que convirtió las propuestas de regiones en parte de la misma red entrenada en lugar de un paso aparte. LibreYOLO incluye un port de la implementación de torchvision para detección."
keywords: [Faster R-CNN, "detección de objetos python", "red de propuestas de regiones", "detector de dos etapas", "Faster R-CNN pytorch", torchvision]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFasterRCNNl.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

Faster R-CNN no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` e `iou` fijan los
umbrales de confianza y de NMS; Faster R-CNN mantiene su paso de NMS upstream,
a diferencia de un detector basado en queries. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Cuatro tamaños, cada uno una configuración distinta de torchvision en lugar de
una versión escalada de la misma: `n` es MobileNetV3-Large con una entrada de
320 px, `s` es el mismo backbone a 800 px, `m` es ResNet-50 con una pirámide de
características, y `l` es la revisión v2, con una cabeza de propuestas de
regiones más profunda y una cabeza de cajas de cuatro convoluciones en lugar de
la de `m`. `n` y `s` sacrifican precisión a cambio de un backbone más ligero.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Faster R-CNN solo exporta a ONNX, con tamaño de batch 1. El grafo exportado
mantiene dentro el paso de redimensionado upstream, así que LibreYOLO fuerza
`dynamic=True` independientemente de lo que se le pase, para que el grafo siga
siendo válido con fuentes que no sean cuadradas. Un archivo `.onnx` exportado
se vuelve a cargar con `LibreYOLO()` según la extensión de su archivo y devuelve
el mismo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
