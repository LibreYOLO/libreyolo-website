---
title: FCOS
families:
  - fcos
seo_title: 'FCOS in LibreYOLO: Vorhersage, Validierung und Export'
description: >-
  Führe FCOS in LibreYOLO für ankerfreie Objekterkennung aus. Installiere, sage
  vorher, validiere und exportiere die torchvision-Portierung mit ResNet-50/FPN
  unter BSD-3-Clause.
lead: >-
  FCOS erkennt Objekte pixelweise, statt eine Menge vordefinierter Anchor Boxes
  zu verwenden. An jeder Position der Feature Map sagt es eine Box und einen
  Centerness-Score vorher. LibreYOLO portiert die torchvision-Implementierung
  für die Objekterkennung.
keywords:
  - fcos python
  - ankerfreie objekterkennung
  - object detection
  - einstufiger detektor
  - torchvision fcos
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreFCOSr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Installation

FCOS benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. Wenn du das Modell ohne Schwellenwertargumente aufrufst, werden die
veröffentlichten Standardwerte von FCOS angewendet: `conf=0.2`, `iou=0.6` und
`max_det=100`. Übergib einen der drei Werte, um ihn zu überschreiben. FCOS
behält einen abschließenden NMS-Schritt über seine pixelweisen Vorhersagen bei.
Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

Es gibt eine Größe: ResNet-50 mit einer Feature Pyramid. Dies ist die einzige
von dieser Familie erkannte Variante.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

FCOS lässt sich nach ONNX, TorchScript und OpenVINO exportieren. FCOS behält
das Seitenverhältnis der Quelle bei, bevor der Graph ausgeführt wird. Für die
ONNX- und OpenVINO-Pfade erzwingt LibreYOLO daher unabhängig vom übergebenen
Wert `dynamic=True`, damit der Graph für aufgefüllte Eingabeformen gültig
bleibt. Eine exportierte `.onnx`-Datei wird anhand ihrer Endung wieder über
`LibreYOLO()` geladen und gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
