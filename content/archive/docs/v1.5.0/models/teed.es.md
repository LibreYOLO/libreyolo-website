---
title: TEED
families:
  - teed
seo_title: 'TEED: detección de bordes con tu propio checkpoint'
description: >-
  Usa TEED en LibreYOLO para predecir un mapa denso de probabilidad de borde.
  Convierte un checkpoint que tengas licenciado y luego predice, valida y
  expórtalo.
lead: >-
  TEED (Tiny and Efficient Edge Detector) es una red convolucional pequeña que
  predice un mapa denso de probabilidad de borde a partir de una sola imagen
  RGB. LibreYOLO incluye su arquitectura solo para detección de bordes; la
  biblioteca no distribuye ningún checkpoint.
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - detección de bordes python
  - BIPED
  - detector de bordes ligero deep learning
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)        # (H, W) float32 en [0, 1]
        print(edges.binary(0.5).sum())  # nº de píxeles de borde tras el umbral
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # F-measure a escala óptima de dataset
        print(metrics["metrics/OIS"])   # F-measure a escala óptima de imagen
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Usar el archivo exportado
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Instalación

TEED no necesita ningún extra opcional. Todo lo que importa está en la
instalación base.

```bash
pip install libreyolo
```

## Predicción

LibreYOLO no distribuye ningún checkpoint de TEED. Los pesos publicados
oficialmente están entrenados sobre BIPED, cuyos términos de dataset publicados
restringen el uso a fines no comerciales, así que LibreYOLO no los replica.
Convierte un checkpoint que tengas licencia para usar con
`weights/convert_teed_weights.py`, que comprueba las claves de los tensores
contra la arquitectura del runtime antes de escribir un archivo que LibreYOLO
puede cargar directamente:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` contiene el resultado: un array float32 `(H, W)` en `[0, 1]`,
con `.binary(threshold)` devolviendo una máscara booleana de bordes. No hay
boxes, así que `conf`, `iou` y `max_det` no tienen ningún efecto. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

TEED incluye un único tamaño en LibreYOLO. El sistema de benchmarks de
LibreYOLO no ha medido esta familia, así que no hay números publicados con los
que compararla.

## Validación

`val()` informa de las F-measures ODS y OIS al estilo de BSDS contra un dataset
emparejado de bordes: imágenes junto a mapas de bordes con el mismo nombre base,
con una máscara de validez opcional para que los píxeles de padding nunca
cuenten. `imgsz` debe ser divisible por el stride de downsample de la red, y
LibreYOLO lanza un error claro si no lo es.

<code-tabs name="val" />

## Exportación

<export-matrix />

La exportación de bordes usa un contrato de runtime de resolución fija y
batch 1: se rechazan `dynamic` y cualquier `batch` distinto de 1, y el grafo
exportado devuelve un único mapa de probabilidad fusionado. Un artefacto
exportado se vuelve a cargar con `LibreYOLO()` según su extensión de archivo,
así que un archivo `.onnx` se comporta como un checkpoint y devuelve el mismo
`Results`.

<code-tabs name="export" />

## Licencia

<provenance-box>

LibreYOLO no publica ningún checkpoint de TEED. No hay nada replicado bajo la
organización LibreYOLO; convierte tú mismo un checkpoint del que tengas licencia
con `weights/convert_teed_weights.py`.

</provenance-box>

## Cita

<citation-block />
