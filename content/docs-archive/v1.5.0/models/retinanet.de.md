---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet in LibreYOLO: Vorhersage, Validierung und Export'
description: >-
  Führe RetinaNet in LibreYOLO für einstufige Objekterkennung mit Focal Loss
  aus. Installiere, sage vorher, validiere und exportiere die
  torchvision-Portierung unter BSD-3-Clause.
lead: >-
  RetinaNet ist ein einstufiger Detektor, der mit Focal Loss trainiert wird.
  Dieser schwächt einfache Negative ab, sodass ein dichtes Anchor-Raster auch
  ohne getrennte Vorschlagsphase seine Accuracy behält. LibreYOLO portiert die
  torchvision-Implementierung für die Objekterkennung.
keywords:
  - retinanet python
  - focal loss
  - objekterkennung
  - einstufiger detektor
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
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreRetinaNetr50v2.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Installation

RetinaNet benötigt kein optionales Zusatzpaket. Alle Importe sind in der
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
RetinaNet behält seinen Upstream-NMS-Schritt über dem dichten Anchor-Raster bei.
Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

Es gibt zwei Größen, beide mit ResNet-50 und einer Feature Pyramid. `r50`
verwendet den ursprünglichen Head. `r50v2` ersetzt ihn durch einen
GroupNorm-Head und einen breiteren P6-Block, der aus der letzten Stufe des
Backbones statt aus der FPN-Ausgabe gespeist wird.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

RetinaNet lässt sich nur nach ONNX und mit Batch-Größe 1 exportieren. Es
skaliert auf eine variable Eingabe unter Beibehaltung des Seitenverhältnisses.
LibreYOLO erzwingt daher unabhängig vom übergebenen Wert `dynamic=True`, damit
der Graph für Quellen verschiedener Formen gültig bleibt. Eine exportierte
`.onnx`-Datei wird anhand ihrer Endung wieder über `LibreYOLO()` geladen und
gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>
