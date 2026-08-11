---
title: YOLOv2
families:
  - yolo2
seo_title: 'YOLOv2 in LibreYOLO: Vorhersage, Validierung und Export'
description: >-
  Führe YOLOv2 (YOLO9000) in LibreYOLO aus: eine eingefrorene Museumsfamilie nur
  für die Inferenz. Vorhersage, Validierung und Export unter einer
  Public-Domain-Lizenz.
lead: >-
  YOLOv2, auch als YOLO9000 veröffentlicht, ist der Darknet-19-Detektor, der
  Anchor-Boxen und eine Passthrough-Schicht in die YOLO-Reihe einführte.
  LibreYOLO bewahrt ihn als eingefrorenes Exponat nur für die Inferenz.
keywords:
  - YOLOv2
  - YOLO9000
  - Darknet
  - Darknet-19
  - Objekterkennung
  - Anchor-Boxen
  - historisches YOLO-Modell
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO2b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO2b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO2b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO2b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO2b.pt format=onnx
        libreyolo export model=LibreYOLO2b.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreYOLO2b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: ba2884a2f6e1b0da
---

## Installation

YOLOv2 benötigt neben dem Basispaket kein zusätzliches Extra.

```bash
pip install libreyolo
```

## Vorhersage

Diese Familie ist ausschließlich für die Inferenz vorgesehen. `train()` löst `NotImplementedError` aus, daher enthält diese Seite keinen Abschnitt zum Training. Vorhersage, Validierung und Export werden unterstützt. Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem jeder anderen Familie, sodass der Wechsel zu einem anderen Detektor nur eine einzige Codezeile erfordert. `conf` filtert anhand des Konfidenzschwellenwerts, `iou` anhand des NMS-Schwellenwerts. Beide werden auf die Anchor-basierten Vorhersagen des `region`-Kopfes angewendet. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall, mAP 50 und mAP 50-95 zurück, gemessen anhand jedes Datensatzes im für die Validierung verwendeten Format.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Du kannst den Graphen auch in einer eigenständigen Laufzeitumgebung ohne installiertes LibreYOLO ausführen. Dann musst du Vor- und Nachverarbeitung selbst implementieren.

<code-tabs name="export" />

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>
