---
title: YOLOv2
families: [yolo2]
seo_title: "YOLOv2 en LibreYOLO: predecir, validar, exportar"
description: "Ejecuta YOLOv2 (YOLO9000) en LibreYOLO: una familia de museo congelada y solo de inferencia. Predice, valida y exporta, bajo una licencia de dominio público."
lead: "YOLOv2, publicado también como YOLO9000, es el detector Darknet-19 que introdujo las anchor boxes y una capa passthrough en la línea YOLO. LibreYOLO lo mantiene como una pieza de museo congelada y solo de inferencia."
keywords: [YOLOv2, YOLO9000, Darknet, Darknet-19, "detección de objetos", "anchor boxes", "yolov2 python", "ejecutar yolov2", "exportar yolov2 onnx", "modelos yolo antiguos"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO2b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO2b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO2b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO2b.pt format=onnx
        libreyolo export model=LibreYOLO2b.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según el sufijo del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreYOLO2b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

YOLOv2 no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Esta familia es solo de inferencia: `train()` lanza `NotImplementedError`, así
que esta página no tiene sección de entrenamiento. La predicción, la validación
y la exportación sí están soportadas. Los pesos se descargan de Hugging Face en
el primer uso y se guardan en la caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelve cada familia, así que
cambiar a otro detector es un cambio de una línea. `conf` filtra el umbral de
confianza e `iou` el umbral de NMS, aplicados sobre las predicciones basadas en
anchors de la cabeza `region`. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que validas.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar a través de `LibreYOLO()` por su
sufijo de archivo, así que un archivo `.onnx` o `.engine` se comporta como un
checkpoint y devuelve el mismo `Results`. Ejecutar el grafo en un runtime
desnudo, sin LibreYOLO instalado, también está soportado, pero entonces el
preprocesamiento y el postprocesamiento corren de tu cuenta.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>
