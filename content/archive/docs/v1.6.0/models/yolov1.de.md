---
title: YOLOv1
families:
  - yolo1
seo_title: 'YOLOv1 in LibreYOLO: Vorhersage, Validierung und Export'
description: >-
  Führe den ursprünglichen YOLOv1-Detektor in LibreYOLO aus: eine eingefrorene
  Museumsfamilie nur für die Inferenz. Vorhersage, Validierung und Export unter
  einer Public-Domain-Lizenz.
lead: >-
  YOLOv1 ist der ursprüngliche Detektor von 2016, der der YOLO-Familie ihren
  Namen gab. Ein konvolutionales Netzwerk mit vollständig verbundenem Kopf sagt
  alle Begrenzungsrahmen und Klassenbewertungen in einem einzigen Durchlauf und
  ohne Anchor-Boxen vorher. LibreYOLO bewahrt ihn als eingefrorenes Exponat nur
  für die Inferenz.
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - Objekterkennung
  - Pascal VOC
  - historisches YOLO-Modell
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreYOLO1b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## Installation

YOLOv1 benötigt neben dem Basispaket kein zusätzliches Extra.

```bash
pip install libreyolo
```

## Vorhersage

Diese Familie ist ausschließlich für die Inferenz vorgesehen. `train()` löst `NotImplementedError` aus, daher enthält diese Seite keinen Abschnitt zum Training. Vorhersage, Validierung und Export werden unterstützt. Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem jeder anderen Familie, sodass der Wechsel zu einem anderen Detektor nur eine einzige Codezeile erfordert. Zwei Eigenschaften sind für diese Familie spezifisch. Der veröffentlichte Checkpoint wurde auf Pascal VOC (2007+2012) und nicht auf COCO trainiert. `box.cls` verweist daher auf die 20 VOC-Kategorien (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor) statt auf die 80 COCO-Kategorien. Außerdem akzeptiert der vollständig verbundene Erkennungskopf jeweils nur ein Bild. Eine Liste von Quellen wird deshalb nacheinander verarbeitet und nicht als echter Batch ausgeführt. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall, mAP 50 und mAP 50-95 zurück. Gemessen wird anhand eines Datensatzes im selben VOC-artigen Labelraum, auf dem der Checkpoint trainiert wurde.

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
