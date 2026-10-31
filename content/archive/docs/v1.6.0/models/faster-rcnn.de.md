---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'Faster R-CNN in LibreYOLO: Vorhersage, Validierung und Export'
description: >-
  Führe Faster R-CNN in LibreYOLO für die Objekterkennung mit vier Backbones
  aus. Installiere, sage vorher, validiere und exportiere die
  torchvision-Portierung unter BSD-3-Clause.
lead: >-
  Faster R-CNN erkennt Objekte mit einem Region Proposal Network, das einen
  zweistufigen Klassifikator speist. Diese Architektur machte Regionsvorschläge
  zu einem Bestandteil desselben trainierten Netzwerks statt zu einem separaten
  Schritt. LibreYOLO portiert die torchvision-Implementierung für die
  Objekterkennung.
keywords:
  - faster r-cnn python
  - objekterkennung
  - region proposal network
  - zweistufiger detektor
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreFasterRCNNl.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Installation

Faster R-CNN benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. `conf` und `iou` legen die Schwellenwerte für Confidence und NMS fest.
Im Gegensatz zu einem Query-basierten Detektor behält Faster R-CNN seinen
Upstream-NMS-Schritt bei. Unter [Vorhersage](/docs/predict) findest du Quellen,
Streaming und die Verarbeitung von Ergebnissen.

## Varianten

Es gibt vier Größen. Jede ist eine andere torchvision-Konfiguration und keine
skalierte Version derselben Konfiguration: `n` verwendet MobileNetV3-Large bei
einer Eingabe von 320 px, `s` dasselbe Backbone bei 800 px, `m` ResNet-50 mit
einer Feature Pyramid und `l` die v2-Überarbeitung. Letztere ersetzt den Head
von `m` durch einen tieferen Region-Proposal-Head und einen Box-Head mit vier
Faltungen. `n` und `s` tauschen Accuracy gegen ein leichteres Backbone.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

Faster R-CNN lässt sich nur nach ONNX und mit Batch-Größe 1 exportieren. Der
exportierte Graph enthält weiterhin den Upstream-Schritt zur Größenänderung.
LibreYOLO erzwingt daher unabhängig vom übergebenen Wert `dynamic=True`, damit
der Graph für nicht quadratische Quellen gültig bleibt. Eine exportierte
`.onnx`-Datei wird anhand ihrer Endung wieder über `LibreYOLO()` geladen und
gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
