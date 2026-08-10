---
title: SSD
families: [ssd]
seo_title: "SSD (SSD300): detección de objetos en LibreYOLO"
description: "Ejecuta SSD300 en LibreYOLO: un detector single-shot con VGG16 para predicción, validación y exportación a ONNX bajo licencia BSD-3-Clause. Sin ruta de entrenamiento."
lead: "SSD (Single Shot MultiBox Detector) predice todas las cajas y las puntuaciones de clase a partir de una rejilla densa de cajas por defecto en una sola pasada hacia delante, sin una etapa aparte de propuesta de regiones. LibreYOLO incluye el checkpoint SSD300 con backbone VGG16 como detector solo de inferencia."
keywords: [SSD, SSD300, "Single Shot MultiBox Detector", "detección de objetos python", VGG16, "detector basado en anchors", "exportar ssd onnx"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSSD300.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")

        # imgsz se omite aquí a propósito: SSD300 se traza en el lienzo nativo
        # de su checkpoint, y cualquier otro valor lanza un error antes de que
        # empiece la exportación.
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

SSD no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. SSD decodifica su rejilla
de cajas por defecto con puntuaciones por clase y después aplica non-maximum
suppression, así que aquí `conf`, `iou` y `max_det` tienen todos un efecto real,
a diferencia de los detectores basados en queries de esta biblioteca. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

SSD incluye un único checkpoint: la red SSD300 con backbone VGG16 en su lienzo
nativo fijo. En esta familia no hay elección de tamaño ni de escala; la
predicción, la validación y la exportación usan todas ese mismo grafo.

El archivo de pesos es `LibreSSD300.pt`, el prefijo de la familia seguido de su
única clave de tamaño, `"300"`. La clase que hay detrás es `LibreSSD`, así que
una construcción directa es `LibreSSD(size="300")` y no una clase con el nombre
del archivo.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

SSD solo exporta a ONNX; el resto de formatos están bloqueados actualmente para
esta familia. La exportación usa siempre el lienzo nativo del checkpoint, y el
grafo expone la cabeza empaquetada en bruto de SSD en lugar de una salida con
non-maximum suppression fusionada, así que `nms=True` no se acepta en el momento
de exportar. Los propios backends de LibreYOLO ejecutan el paso de decodificado
y supresión después de volver a cargar el grafo.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

El código de SSD300 de LibreYOLO no está portado de la propia versión en Caffe
de los autores del paper; deriva de la implementación de SSD300 de torchvision,
con licencia BSD-3-Clause, y ese es el repositorio enlazado arriba como fuente
upstream. Los pesos VGG16 del backbone se remontan además a la VGGNet reducida
totalmente convolucional de Oxford, publicada bajo CC BY 4.0 por Karen Simonyan
y Andrew Zisserman.

</provenance-box>

## Cita

<citation-block />
