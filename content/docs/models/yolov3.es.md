---
title: YOLOv3
families: [yolo3]
seo_title: "YOLOv3 en LibreYOLO: predecir, validar y exportar"
description: "Ejecuta YOLOv3 en LibreYOLO: una familia de museo congelada y solo de inferencia, con tamaños tiny, base y SPP. Predice, valida y exporta, bajo una licencia de dominio público."
lead: "YOLOv3 es el detector Darknet-53 que añadió la predicción multiescala y los clasificadores logísticos independientes a la línea YOLO. LibreYOLO lo conserva como una pieza de museo congelada y solo de inferencia, en tamaños tiny, base y SPP."
keywords: [YOLOv3, Darknet, Darknet-53, "detección de objetos", "detección multiescala", "yolov3 python", "yolov3 pesos preentrenados", "darknet yolov3", "modelos yolo antiguos"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO3b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: Tamaño SPP
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La variante SPP añade un bloque de spatial pyramid pooling antes de las
        # cabezas de detección y funciona con su propio tamaño de entrada nativo.
        model = LibreYOLO("LibreYOLO3spp.pt")
        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según el sufijo del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto Results.
        model = LibreYOLO("LibreYOLO3b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

YOLOv3 no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Esta familia es solo de inferencia: `train()` lanza `NotImplementedError`, así
que esta página no tiene sección de entrenamiento. Predicción, validación y
exportación sí están soportadas. Los pesos se descargan de Hugging Face en el
primer uso y se guardan en la caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelve cada familia, así que
cambiar a otro detector es un cambio de una línea. `conf` filtra el umbral de
confianza e `iou` el umbral de NMS, aplicados por escala antes de fusionar los
boxes de las tres cabezas. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que valides.

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
