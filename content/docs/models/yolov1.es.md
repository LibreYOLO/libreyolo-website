---
title: YOLOv1
families: [yolo1]
seo_title: "YOLOv1 en LibreYOLO: predicción, validación y exportación"
description: "Ejecuta el detector YOLOv1 original en LibreYOLO: una familia de museo congelada y solo de inferencia. Predicción, validación y exportación, con licencia de dominio público."
lead: "YOLOv1 es el detector original de 2016 que dio nombre a la familia YOLO: una sola red convolucional con una cabeza totalmente conectada predice todos los bounding boxes y puntuaciones de clase en una única pasada, sin anchor boxes. LibreYOLO lo incluye como una pieza de museo congelada y solo de inferencia."
keywords: [YOLOv1, YOLO v1, Darknet, "detección de objetos python", Pascal VOC, "familia de museo"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO1b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto
        # Results.
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

YOLOv1 no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Esta familia es solo de inferencia: `train()` lanza `NotImplementedError`, así
que esta página no tiene sección de entrenamiento. La predicción, la validación
y la exportación sí están soportadas. Los pesos se descargan de Hugging Face en
el primer uso y se guardan en la caché local.

<code-tabs name="predict" />

El objeto `Results` que se devuelve es el mismo que devuelve cualquier familia,
así que cambiar a otro detector es un cambio de una línea. Dos cosas son
específicas de esta familia. El checkpoint publicado está entrenado con Pascal
VOC (2007+2012), no con COCO, así que `box.cls` indexa las 20 categorías de VOC
(aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train,
tvmonitor) en lugar de las 80 de COCO. Y la cabeza de detección totalmente
conectada acepta una imagen cada vez, así que una lista de fuentes se recorre en
bucle en vez de ejecutarse como un batch real. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren la precisión, el
recall, el mAP 50 y el mAP 50-95, medidos sobre un dataset en el mismo espacio
de etiquetas estilo VOC con el que se entrenó el checkpoint.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su extensión
de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Ejecutar el grafo en un runtime
pelado, sin LibreYOLO instalado, también está soportado, pero entonces el
preprocesado y el postprocesado corren de tu cuenta.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>
