---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: predice y exporta bajo Apache-2.0'
description: >-
  Usa LW-DETR en LibreYOLO para detección de objetos en tiempo real. Instala,
  predice, valida y exporta cinco tamaños basados en ViT, todos con licencia
  Apache-2.0.
lead: >-
  Un transformer de detección con ViT plano que Baidu planteó como alternativa
  en tiempo real a los detectores YOLO. LibreYOLO incluye cinco tamaños para
  detección, solo inferencia.
keywords:
  - LW-DETR
  - transformer de detección
  - detección de objetos en tiempo real
  - ViT plano
  - DETR
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() devuelve un dict normal, no un objeto
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Usar el archivo exportado
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # La factoría enruta según el sufijo del archivo, así que un artefacto

        # exportado se carga como cualquier checkpoint y devuelve el mismo
        objeto Results.

        model = LibreYOLO("LibreLWDETRt.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Instalación

LW-DETR no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local.

<code-tabs name="predict" />

El objeto `Results` devuelto es el mismo que devuelven todas las familias, así
que cambiar a otro detector es un cambio de una línea. `conf` y `max_det`
filtran la selección de queries; `iou` se acepta por paridad de API pero no
tiene efecto, porque el decoder es un predictor de conjuntos sin paso de NMS.
Consulta [predicción](/docs/predict) para fuentes, streaming y manejo de
resultados.

LW-DETR es solo de inferencia en LibreYOLO. Upstream lo entrena con supervisión
uno-a-muchos de Group-DETR sobre varios grupos de queries y una loss de
clasificación consciente del IoU; esa receta no está implementada aquí, así que
`train()` lanza `NotImplementedError`.

## Variantes

Cinco tamaños, todos con el mismo encoder de ViT plano, proyector multiescala y
decoder de DETR deformable, y todos funcionando a la misma resolución de
entrada. Los dos más pequeños comparten el ancho del encoder y se diferencian
por la profundidad en bloques; los dos siguientes comparten un encoder más ancho
y se diferencian por cuántos niveles del proyector alimentan al decoder; el más
grande sube al encoder más ancho de todos.

## Validación

`val()` devuelve un diccionario de claves `metrics/` que cubren precisión,
recall, mAP 50 y mAP 50-95, medidas contra cualquier dataset en el formato con
el que entrenaste.

<code-tabs name="val" />

## Exportación

<export-matrix />

Un artefacto exportado se vuelve a cargar con `LibreYOLO()` según su sufijo de
archivo, así que un archivo `.onnx` o `.engine` se comporta como un checkpoint y
devuelve el mismo `Results`. [Exportación](/docs/export) lista los argumentos que
acepta cada formato.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
