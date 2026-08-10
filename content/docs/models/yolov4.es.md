---
title: YOLOv4
families: [yolo4]
seo_title: "YOLOv4: ejecutar, validar y exportar en LibreYOLO"
description: "Ejecuta YOLOv4 en LibreYOLO: una familia de museo congelada y solo de inferencia, con backbone CSPDarknet-53. Predice, valida y exporta, con licencia de dominio público."
lead: "YOLOv4 combina un backbone CSPDarknet-53, un bloque SPP y un neck PANet con activaciones Mish. LibreYOLO lo incluye como una pieza de museo congelada y solo de inferencia, en tamaños tiny y base."
keywords: [YOLOv4, Darknet, CSPDarknet-53, PANet, "detección de objetos python", "activación Mish", "familia de museo"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO4b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO4b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO4b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO4b.pt format=onnx
        libreyolo export model=LibreYOLO4b.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto
        # Results.
        model = LibreYOLO("LibreYOLO4b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

YOLOv4 no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Esta familia es solo de inferencia: `train()` lanza `NotImplementedError`, así
que esta página no tiene sección de entrenamiento. La predicción, la validación
y la exportación sí están soportadas. Los pesos se descargan de Hugging Face en
el primer uso y se guardan en la caché local.

<code-tabs name="predict" />

El objeto `Results` que se devuelve es el mismo que devuelven todas las
familias, así que cambiar a otro detector es un cambio de una línea. `conf`
filtra el umbral de confianza e `iou` el umbral de NMS, aplicados después del
escalado de centros `scale_x_y` propio de cada cabeza. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que valides.

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

## Cita

<citation-block />
