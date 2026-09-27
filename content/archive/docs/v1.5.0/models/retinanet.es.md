---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet en LibreYOLO: predecir, validar y exportar'
description: >-
  Ejecuta RetinaNet en LibreYOLO para detección de objetos de una etapa con
  focal loss. Instala, predice, valida y exporta el port de torchvision con
  licencia BSD-3-Clause.
lead: >-
  RetinaNet es un detector de una etapa entrenado con focal loss, que reduce el
  peso de los negativos fáciles para que una rejilla densa de anchors ya no
  necesite una etapa de propuestas aparte para mantener la precisión. LibreYOLO
  porta la implementación de torchvision para detección.
keywords:
  - RetinaNet
  - focal loss
  - detección de objetos python
  - detector de una etapa
  - retinanet pytorch
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un
        # artefacto exportado se carga como cualquier checkpoint y devuelve el
        # mismo objeto Results.
        model = LibreYOLO("LibreRetinaNetr50v2.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Instalación

RetinaNet no necesita ningún extra opcional. Todo lo que importa está en la
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
umbrales de confianza y de NMS; RetinaNet mantiene su paso de NMS original
sobre la rejilla densa de anchors. Consulta [predicción](/docs/predict) para
fuentes, streaming y manejo de resultados.

## Variantes

Dos tamaños, ambos ResNet-50 con una pirámide de características: `r50` es la
cabeza original, y `r50v2` la sustituye por una cabeza con GroupNorm y un
bloque P6 más ancho alimentado desde la última etapa del backbone en lugar de
desde la salida del FPN.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

RetinaNet solo exporta a ONNX, con tamaño de batch 1. RetinaNet redimensiona a
una entrada variable que conserva la relación de aspecto, así que LibreYOLO
fuerza `dynamic=True` independientemente de lo que se pase, para mantener el
grafo válido con fuentes de formas distintas. Un archivo `.onnx` exportado se
vuelve a cargar con `LibreYOLO()` según la extensión del archivo y devuelve el
mismo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>
