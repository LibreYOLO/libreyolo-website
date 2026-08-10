---
title: FCOS
families: [fcos]
seo_title: "FCOS en LibreYOLO: predecir, validar y exportar"
description: "Ejecuta FCOS en LibreYOLO para detección de objetos sin anchors. Instala, predice, valida y exporta el port de torchvision con licencia BSD-3-Clause, ResNet-50/FPN."
lead: "FCOS detecta objetos píxel a píxel en lugar de apoyarse en un conjunto de anchor boxes predefinidos, y predice un box y una puntuación de centerness en cada posición del mapa de características. LibreYOLO porta la implementación de torchvision para detección."
keywords: [FCOS, "detector anchor-free", "detección de objetos python", "detección sin anchors", "detector de una etapa", torchvision]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFCOSr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve el
        # mismo objeto Results.
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Instalación

FCOS no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la
caché local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. Llamar al modelo sin
argumentos de umbral aplica los valores por defecto publicados por el propio
FCOS, `conf=0.2`, `iou=0.6` y `max_det=100`; pasa cualquiera de los tres para
sobrescribirlos. FCOS mantiene un paso final de NMS sobre sus predicciones por
píxel. Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

## Variantes

Un solo tamaño: ResNet-50 con una pirámide de características, la única
variante que reconoce esta familia.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

FCOS exporta a ONNX, TorchScript y OpenVINO. FCOS conserva la relación de
aspecto de la fuente antes de que se ejecute el grafo, así que LibreYOLO fuerza
`dynamic=True` en las rutas de ONNX y OpenVINO independientemente de lo que se
pase, para mantener el grafo válido con formas de entrada con padding. Un
archivo `.onnx` exportado se vuelve a cargar con `LibreYOLO()` según la
extensión del archivo y devuelve el mismo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
