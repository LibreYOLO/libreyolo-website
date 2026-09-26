---
title: TEED
families:
  - teed
seo_title: 'TEED: Kantenerkennung mit eigenem Checkpoint'
description: >-
  Verwende TEED in LibreYOLO für die Vorhersage dichter
  Kantenwahrscheinlichkeiten. Konvertiere einen lizenzierten Checkpoint und
  nutze ihn für Vorhersage, Validierung und Export.
lead: >-
  TEED (Tiny and Efficient Edge Detector) ist ein kleines konvolutionales
  Netzwerk, das aus einem RGB-Bild eine dichte Karte von
  Kantenwahrscheinlichkeiten vorhersagt. LibreYOLO bindet seine Architektur
  ausschließlich für die Kantenerkennung ein. Die Bibliothek enthält keinen
  Checkpoint.
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - Kantenerkennung
  - BIPED
  - dichte Vorhersage
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)        # (H, W) float32 im Bereich [0, 1]

        print(edges.binary(0.5).sum())  # Anzahl der Kantenpixel nach
        Schwellenwertanwendung
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        metrics = model.val(data="my-dataset.yaml", imgsz=352)


        print(metrics["metrics/ODS"])   # F-Maß mit optimalem Schwellenwert auf
        Datensatzebene

        print(metrics["metrics/OIS"])   # F-Maß mit optimalem Schwellenwert auf
        Bildebene
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
    - label: Exportierte Datei verwenden
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Installation

TEED benötigt kein optionales Extra. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

LibreYOLO stellt keinen TEED-Checkpoint bereit. Die offiziell veröffentlichten Gewichte wurden auf BIPED trainiert, dessen publizierte Datensatzbedingungen die Nutzung auf nicht kommerzielle Zwecke beschränken. Daher spiegelt LibreYOLO diese Gewichte nicht. Konvertiere einen Checkpoint, für dessen Nutzung du eine Lizenz besitzt, mit `weights/convert_teed_weights.py`. Das Skript prüft die Tensorschlüssel gegen die Laufzeitarchitektur, bevor es eine Datei schreibt, die LibreYOLO direkt laden kann:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` enthält das Ergebnis als float32-Array der Form `(H, W)` mit Werten in `[0, 1]`. `.binary(threshold)` gibt eine boolesche Kantenmaske zurück. Es gibt keine Boxen, daher haben `conf`, `iou` und `max_det` keine Wirkung. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

LibreYOLO bietet TEED in einer Größe an. Die Benchmark-Infrastruktur von LibreYOLO hat diese Familie noch nicht gemessen, daher gibt es keine veröffentlichten Vergleichswerte.

## Validierung

`val()` meldet ODS- und OIS-F-Maße im BSDS-Stil für einen gekoppelten Kantendatensatz. Dabei liegen Bilder neben gleichnamigen Kantenkarten. Eine optionale Gültigkeitsmaske verhindert, dass aufgefüllte Pixel gezählt werden. `imgsz` muss durch den Downsampling-Stride des Netzwerks teilbar sein. Andernfalls löst LibreYOLO einen verständlichen Fehler aus.

<code-tabs name="val" />

## Export

<export-matrix />

Der Kantenexport verwendet einen Laufzeitvertrag mit fester Auflösung und Batchgröße 1. `dynamic` und ein anderer Wert für `batch` als 1 werden abgelehnt. Der exportierte Graph gibt eine einzige zusammengeführte Wahrscheinlichkeitskarte aus. Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen, sodass sich eine `.onnx`-Datei wie ein Checkpoint verhält und dasselbe `Results`-Objekt zurückgibt.

<code-tabs name="export" />

## Lizenzierung

<provenance-box>

LibreYOLO veröffentlicht keinen TEED-Checkpoint. Innerhalb der LibreYOLO-Organisation werden keine Gewichte gespiegelt. Konvertiere stattdessen einen Checkpoint, für den du eine Lizenz besitzt, mit `weights/convert_teed_weights.py`.

</provenance-box>

## Zitieren

<citation-block />
