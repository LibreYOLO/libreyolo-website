---
title: PP-OCRv5
families: [ppocr]
seo_title: "PP-OCRv5: detección y reconocimiento de texto en LibreYOLO"
description: "Usa PP-OCRv5 en LibreYOLO para OCR multilingüe de texto en escenas. Instala, predice y valida los checkpoints t y l, con licencia Apache-2.0."
lead: "PP-OCRv5 es el pipeline de detección y reconocimiento de texto de PaddleOCR: un detector de binarización diferenciable localiza los cuadriláteros de texto y un reconocedor SVTR/CTC los lee. LibreYOLO lo porta a PyTorch en dos niveles."
keywords: [PP-OCRv5, PaddleOCR, OCR, "detección de texto en imágenes", "reconocimiento de texto", "extraer texto de una imagen python"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Cuadriláteros
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # polígonos (N, 4, 2) en orden de lectura: arriba-izquierda,
        # arriba-derecha, abajo-derecha, abajo-izquierda. Los cuadriláteros de
        # detección son polígonos reales (texto rotado), así que rellenan
        # result.ocr, no result.boxes.
        print(result.ocr.data.shape)
        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # métrica principal
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
---

## Instalación

PP-OCRv5 no necesita ningún extra más allá del paquete base.

```bash
pip install libreyolo
```

## Predicción

Los pesos se descargan de Hugging Face en el primer uso y quedan cacheados
localmente.

<code-tabs name="predict" />

Cada checkpoint agrupa las dos etapas, detección y reconocimiento, en un único
archivo `.pt`, con el charset de reconocimiento y los valores por defecto del
pipeline guardados en los metadatos del checkpoint. El reconocedor lee chino
simplificado y tradicional, inglés, japonés y pinyin con un único diccionario.
`result.ocr` es un payload `OCRRegions`: `.data` contiene los polígonos de
cuatro puntos, `.texts` las transcripciones, `.conf` la puntuación de
reconocimiento de cada región y `.det_conf` la puntuación de detección. Las
fuentes con varias imágenes se procesan de forma secuencial: el pipeline de dos
etapas no agrupa en batch entre imágenes. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Variantes

Dos niveles: `t`, construido sobre los backbones más ligeros
PP-LCNetV3/PP-OCRv5_mobile para uso en CPU, y `l`, construido sobre backbones de
servidor PP-HGNetV2 para mayor precisión. Ambos niveles ejecutan la detección con
un límite fijo del lado largo y reconocen los recortes en batches; `rec_batch`
controla cuántos recortes pasan por el reconocedor en cada forward pass.

## Validación

`val()` mide el pipeline contra un directorio de imágenes más un archivo
`labels/<split>.jsonl`, o el YAML de dataset equivalente, donde cada etiqueta
enumera los polígonos de las regiones de texto de cada imagen y sus
transcripciones. Informa del hmean de detección (precisión/recall/F1 emparejados
por IoU), el F1 end-to-end (el hmean más una coincidencia exacta de la
transcripción tras normalizarla, la métrica de fitness del checkpoint) y 1-NED,
la distancia de edición normalizada media sobre los pares emparejados.

<code-tabs name="val" />

## Exportación

<export-matrix />

PP-OCRv5 es un pipeline de dos redes, detección y reconocimiento avanzando
juntos, no un único grafo trazable, y la exportación no está implementada para
él: todavía no hay ningún formato soportado. Haz fine-tuning directamente del
código de entrenamiento upstream con licencia Apache-2.0 y convierte el
resultado con `weights/convert_ppocr_weights.py` si necesitas un checkpoint
fuera de este formato.

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box></provenance-box>

## Cita

<citation-block />
