---
title: YOLOv3
families:
  - yolo3
seo_title: 'YOLOv3 in LibreYOLO: Vorhersage, Validierung und Export'
description: >-
  Führe YOLOv3 in LibreYOLO aus: eine eingefrorene Museumsfamilie nur für die
  Inferenz in den Größen Tiny, Base und SPP. Vorhersage, Validierung und Export
  unter einer Public-Domain-Lizenz.
lead: >-
  YOLOv3 ist der Darknet-53-Detektor, der mehrskalige Vorhersagen und
  unabhängige logistische Klassifikatoren in die YOLO-Reihe einführte. LibreYOLO
  bewahrt ihn als eingefrorenes Exponat nur für die Inferenz in den Größen Tiny,
  Base und SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - Objekterkennung
  - mehrskalige Objekterkennung
  - historisches YOLO-Modell
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: SPP-Größe
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die SPP-Variante fügt vor den Erkennungsköpfen einen
        Spatial-Pyramid-Pooling-Block

        # hinzu und wird mit ihrer eigenen nativen Eingabegröße ausgeführt.

        model = LibreYOLO("LibreYOLO3spp.pt")

        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreYOLO3b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Installation

YOLOv3 benötigt neben dem Basispaket kein zusätzliches Extra.

```bash
pip install libreyolo
```

## Vorhersage

Diese Familie ist ausschließlich für die Inferenz vorgesehen. `train()` löst `NotImplementedError` aus, daher enthält diese Seite keinen Abschnitt zum Training. Vorhersage, Validierung und Export werden unterstützt. Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem jeder anderen Familie, sodass der Wechsel zu einem anderen Detektor nur eine einzige Codezeile erfordert. `conf` filtert anhand des Konfidenzschwellenwerts, `iou` anhand des NMS-Schwellenwerts. Die Filterung wird für jede Skala ausgeführt, bevor die Boxen aller drei Köpfe zusammengeführt werden. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

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
