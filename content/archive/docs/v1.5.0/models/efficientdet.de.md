---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: Objekterkennung in LibreYOLO'
description: >-
  Führe EfficientDet D0-D4 in LibreYOLO aus: BiFPN-Detektoren für Vorhersage,
  Validierung und Export nach ONNX, TensorRT und OpenVINO unter Apache-2.0.
lead: >-
  EfficientDet kombiniert ein EfficientNet-Backbone mit einem wiederholten
  bidirektionalen Feature Pyramid Network (BiFPN) und skaliert Tiefe, Breite und
  Auflösung gemeinsam über fünf Größen. LibreYOLO bietet es als reinen
  Inferenzdetektor an.
keywords:
  - efficientdet python
  - bifpn objekterkennung
  - efficientnet detektor
  - object detection
  - compound scaling
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreEfficientDetd0.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Installation

EfficientDet benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. EfficientDet dekodiert Anchor-basierte Kandidaten und führt anschließend
eine klassenweise Non-Maximum Suppression aus. `conf`, `iou` und `max_det`
haben hier daher alle eine tatsächliche Wirkung. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt fünf Größen von D0 bis D4. Jede größere Stufe kombiniert ein größeres
EfficientNet-Backbone mit einem tieferen, breiteren BiFPN und einem tieferen
Vorhersage-Head. Parameteranzahl und Rechenaufwand wachsen somit gemeinsam nach
der Compound-Scaling-Regel der Veröffentlichung.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Die D0-D4-Checkpoints von LibreYOLO werden über das Apache-2.0-Projekt
rwightman/efficientdet-pytorch konvertiert. Dieses spiegelt seinerseits die
offiziellen, mit TensorFlow trainierten Gewichte aus google/automl, ohne die
gelernten Tensoren zu verändern. Es wurde kein Quellcode aus dem LGPL-lizenzierten
Projekt zylo117/Yet-Another-EfficientDet-Pytorch eingesehen oder verwendet.

</provenance-box>
