---
title: Mask R-CNN
families: [mask_rcnn]
seo_title: "Mask R-CNN en LibreYOLO: predecir, validar y exportar"
description: "Ejecuta Mask R-CNN en LibreYOLO para detección de objetos y segmentación de instancias. Instala, predice, valida y exporta el port de torchvision con licencia BSD-3-Clause."
lead: "Mask R-CNN añade a Faster R-CNN una rama de máscaras por región, que predice una máscara de segmentación junto a cada caja que detecta. LibreYOLO incluye un port de la implementación de torchvision para detección y segmentación de instancias."
keywords: [Mask R-CNN, "segmentación de instancias", "detección de objetos python", Faster R-CNN, "Mask R-CNN pytorch", torchvision, "detector de dos etapas"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMaskRCNNr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Solo cajas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" omite la cabeza de máscaras y devuelve las cajas del mismo
        # checkpoint, sin máscaras en el resultado.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # cajas
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
---

## Instalación

Mask R-CNN no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. Cargar el checkpoint sin
el argumento `task` devuelve máscaras de instancia, ya que la segmentación es la
tarea por defecto de esta familia; `result.masks` las lleva entonces junto a las
cajas. Pasar `task="detect"` carga los mismos pesos sin la cabeza de máscaras y
devuelve solo cajas. `conf` e `iou` fijan los umbrales de confianza y de NMS;
Mask R-CNN mantiene su paso de NMS upstream, a diferencia de un detector basado
en queries. Consulta [predicción](/docs/predict) para fuentes, streaming y
manejo de resultados.

## Variantes

Un solo backbone: ResNet-50 con una pirámide de características, usando el
builder v2 de Mask R-CNN de torchvision. El checkpoint publicado lleva licencia
BSD-3-Clause y sirve para las dos tareas de esta familia, así que no hay ningún
tamaño entre el que elegir.

## Validación

`val()` devuelve un diccionario de claves `metrics/`. Con la tarea de
segmentación por defecto de este checkpoint, la clave `metrics/mAP50-95` a secas
contiene la puntuación de las máscaras, y la misma ejecución reporta las cajas
bajo el sufijo `(B)`, así que ambas están disponibles en una sola pasada.

<code-tabs name="val" />

## Exportación

<export-matrix />

Mask R-CNN solo exporta a ONNX, con tamaño de batch 1. El grafo exportado
mantiene dentro los pasos de redimensionado y de pegado de máscaras upstream,
así que LibreYOLO fuerza `dynamic=True` independientemente de lo que se le pase,
para que el grafo siga siendo válido con fuentes que no sean cuadradas. Un
archivo `.onnx` exportado se vuelve a cargar con `LibreYOLO()` según la
extensión de su archivo y devuelve el mismo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia. El único checkpoint de
abajo aparece listado bajo detect, pero el mismo archivo se carga también para
segmentación: no pases el argumento `task` y devolverá máscaras por defecto.

<checkpoint-table />

## Licencia

<provenance-box>

Mask R-CNN está construido como una subclase del wrapper de Faster R-CNN de
LibreYOLO: comparte la misma fuente de torchvision y la misma licencia
BSD-3-Clause, y añade el predictor de máscaras y la cabeza RoI de máscaras del
mismo commit portado.

</provenance-box>
