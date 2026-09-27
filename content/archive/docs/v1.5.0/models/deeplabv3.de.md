---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: semantische ASPP-Segmentierung vorhersagen und exportieren'
description: >-
  Nutze DeepLabv3 in LibreYOLO für die semantische Segmentierung. Installation,
  Vorhersage, Validierung und Export der ResNet- und MobileNetV3-Checkpoints von
  torchvision.
lead: >-
  Ein Netz für semantische Segmentierung, das Features parallel bei mehreren
  Dilatationsraten poolt (Atrous Spatial Pyramid Pooling), bevor es jedes Pixel
  klassifiziert. LibreYOLO liefert es ausschließlich für die semantische
  Segmentierung aus.
keywords:
  - DeepLabv3
  - ASPP
  - atrous spatial pyramid pooling
  - semantische segmentierung python
  - bilder pixelweise segmentieren
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) Klassen-IDs
        print(mask.classes)      # sortierte Klassen-IDs im Bild
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory entscheidet anhand der Dateiendung: ein exportiertes
        # Artefakt lädt wie jeder Checkpoint und liefert dasselbe Results.
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## Installation

DeepLabv3 braucht kein optionales Extra. Alles, was es importiert, steckt in
der Basisinstallation.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert. Das Suffix `-sem` im Dateinamen ist für diese Familie
Pflicht.

<code-tabs name="predict" />

Die semantische Segmentierung liefert eine Klassen-ID pro Pixel und keine
Boxen, deshalb trägt `result.semantic_mask` ein `(H, W)`-Array in `.data` und
die Liste der im Bild vorhandenen Klassen-IDs in `.classes`. `conf`, `iou` und
`max_det` werden aus Gründen der API-Parität akzeptiert, haben aber keine
Wirkung: Das Modell weist jedem Pixel per Argmax eine Klasse zu, ohne
Schwellenwert für die Confidence und ohne NMS-Schritt. Siehe
[Vorhersage](/docs/predict) für Quellen, Streaming und den Umgang mit
Ergebnissen.

## Varianten

Drei Backbones: dilatiertes ResNet-50, dilatiertes ResNet-101 und dilatiertes
MobileNetV3-Large. Das ist DeepLabv3, nicht DeepLabv3+, es gibt also keine
Decoder-Stufe und keine CRF-Verfeinerung, passend zur Implementierung von
torchvision statt zum Referenzcode des Papers selbst.

LibreYOLO trainiert DeepLabv3 nicht: `train()` löst für diese Familie
`NotImplementedError` aus, was die [Support-Stufe](/docs/models) weiter oben
als reine Inferenz kennzeichnet. Die drei veröffentlichten Checkpoints sind
torchvisions eigene Gewichte, auf COCO mit den VOC-Labels trainiert und für den
Loader von LibreYOLO konvertiert.

## Validierung

`val()` liefert `metrics/mIoU` und `metrics/pixel_accuracy`, gemessen an jedem
Datensatz in dem Format, mit dem du trainiert hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt lädt über `LibreYOLO()` anhand seiner Dateiendung
wieder, eine `.onnx`- oder `.engine`-Datei verhält sich also wie ein Checkpoint
und liefert dieselben `Results`. [Export](/docs/export) listet die Argumente
auf, die jedes Format akzeptiert.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei für diese Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>
