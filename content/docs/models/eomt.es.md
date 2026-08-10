---
title: EoMT
families: [eomt]
seo_title: "EoMT: predice segmentación semántica, de instancias y panóptica"
description: "Usa EoMT en LibreYOLO para segmentación semántica, de instancias y panóptica sobre un vision transformer DINOv2 puro, sin decoder. Licencia MIT."
lead: "Una red de segmentación construida sobre un vision transformer puro sin decoder de píxeles dedicado: unas queries aprendidas extra añadidas al propio encoder predicen las máscaras. LibreYOLO la soporta para segmentación semántica, de instancias y panóptica."
keywords: [EoMT, encoder-only mask transformer, DINOv2, "segmentación panóptica python", "segmentación de instancias python", "segmentación semántica python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Semántica
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) ids de clase
        print(mask.classes)      # ids de clase presentes en la imagen, ordenados
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # El sufijo -seg del nombre del archivo selecciona la tarea de
        # instancias, así que aquí no hace falta ningún argumento task.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Panóptica
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) ids de segmento
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-sem.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Semántica
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Segmentación de instancias
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # máscaras
        print(metrics["metrics/mAP50-95(B)"])   # bounding boxes
    - label: Panóptica
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # La factoría enruta según la extensión del archivo, así que un artefacto
        # exportado se carga como cualquier checkpoint y devuelve el mismo objeto
        # Results.
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
---

## Instalación

EoMT no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y se guardan en la caché
local. El sufijo de tarea del nombre del archivo (`-sem`, `-seg`, `-panoptic`)
selecciona la tarea, y `LibreYOLO()` la deduce de ese nombre, así que no hace
falta ningún argumento `task=`.

<code-tabs name="predict" />

La segmentación semántica rellena `result.semantic_mask`, un array `(H, W)` de
ids de clase en `.data`. La segmentación de instancias rellena `result.boxes` y
`result.masks`, con la misma forma que devuelve cualquier otra familia de
segmentación. La segmentación panóptica rellena `result.panoptic`: un mapa de
ids de segmento `(H, W)` en `.data`, más `.segments_info`, una lista de dicts
`{"id", "category_id"}`, uno por segmento. `conf` filtra la selección de
queries; `iou` no tiene ningún efecto en la tarea semántica, porque hace argmax
por píxel sin paso de NMS. Consulta [predicción](/docs/predict) para fuentes,
streaming y manejo de resultados.

## Variantes

Tres tamaños de encoder, s/b/l, todos basados en DINOv2. El checkpoint semántico
está entrenado en ADE20K a 512 px; los checkpoints de instancias y panóptico
están entrenados en COCO a 640 px, con un segundo checkpoint de instancias
entrenado a 1280 px. Upstream solo publica pesos de segmentación de instancias
con DINOv2 en el tamaño l; s y b se publican únicamente para semántica y
panóptica. Existen variantes de EoMT basadas en DINOv3 en upstream, pero no se
distribuyen aquí, porque dependen de pesos DINOv3 restringidos y no comerciales.

LibreYOLO no entrena EoMT: `train()` lanza `NotImplementedError` para esta
familia, algo que el [nivel de soporte](/docs/models) de arriba marca como solo
inferencia.

## Validación

`val()` despacha según la tarea. La semántica devuelve `metrics/mIoU` y
`metrics/pixel_accuracy`. La segmentación de instancias devuelve las mismas
claves de mAP de máscaras y bounding boxes que las demás familias de
segmentación. La panóptica devuelve la Panoptic Quality como `metrics/PQ`,
dividida en `metrics/SQ` (calidad de segmentación) y `metrics/RQ` (calidad de
reconocimiento), más `metrics/PQ_things` y `metrics/PQ_stuff`.

<code-tabs name="val" />

## Exportación

<export-matrix />

Hoy por hoy solo se exporta la tarea semántica: la segmentación de instancias y
la panóptica llaman a `export()` y reciben `NotImplementedError`, porque su
salida de máscaras por query todavía no tiene un contrato de exportación en
runtime. Un artefacto semántico exportado se vuelve a cargar con `LibreYOLO()`
según su extensión de archivo, así que un archivo `.onnx` o `.engine` se comporta
como un checkpoint y devuelve el mismo `Results`.

<code-tabs name="export" />

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
