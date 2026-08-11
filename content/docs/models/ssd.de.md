---
title: SSD
families:
  - ssd
seo_title: 'SSD (SSD300): Objekterkennung in LibreYOLO'
description: >-
  Führe SSD300 in LibreYOLO aus: einen VGG16-Single-Shot-Detektor für
  Vorhersage, Validierung und ONNX-Export unter BSD-3-Clause. Kein
  Trainingspfad.
lead: >-
  SSD (Single Shot MultiBox Detector) sagt alle Begrenzungsrahmen und
  Klassenbewertungen in einem einzigen Vorwärtsdurchlauf aus einem dichten
  Raster von Standardboxen vorher, ohne separate Region-Proposal-Stufe.
  LibreYOLO stellt den VGG16-basierten SSD300-Checkpoint als reinen
  Inferenzdetektor bereit.
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - Objekterkennung
  - VGG16
  - Anchor-basierter Detektor
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSSD300.pt")


        # imgsz wird hier bewusst ausgelassen: SSD300 wird mit der nativen

        # Canvas-Größe seines Checkpoints aufgezeichnet; jeder andere Wert löst
        vor dem Export einen Fehler aus.

        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreSSD300.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## Installation

SSD benötigt kein optionales Extra. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem jeder anderen Familie, sodass der Wechsel zu einem anderen Detektor nur eine einzige Codezeile erfordert. SSD decodiert sein Raster aus Standardboxen mit klassenweisen Bewertungen und führt anschließend Non-Maximum Suppression aus. Deshalb haben `conf`, `iou` und `max_det` hier tatsächlich eine Wirkung, anders als bei den abfragebasierten Detektoren dieser Bibliothek. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

SSD stellt einen Checkpoint bereit: das VGG16-basierte SSD300-Netzwerk mit seiner festen nativen Canvas-Größe. Diese Familie bietet keine Auswahl an Größen oder Skalierungen. Vorhersage, Validierung und Export verwenden denselben Graphen.

Die Gewichtsdatei heißt `LibreSSD300.pt`. Der Name besteht aus dem Familienpräfix und dem einzigen Größenschlüssel `"300"`. Die zugrunde liegende Klasse heißt `LibreSSD`, sodass die direkte Erstellung `LibreSSD(size="300")` lautet und nicht über eine nach der Datei benannte Klasse erfolgt.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall, mAP 50 und mAP 50-95 zurück, gemessen anhand jedes Datensatzes im Format, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

SSD kann nur nach ONNX exportiert werden. Alle anderen Formate sind für diese Familie derzeit gesperrt. Der Export verwendet immer die native Canvas-Größe des Checkpoints. Der Graph gibt den gepackten Rohkopf von SSD aus und keine zusammengeführte Non-Maximum-Suppression-Ausgabe, weshalb `nms=True` beim Export nicht akzeptiert wird. Die LibreYOLO-Backends führen das Decodieren und die Unterdrückung nach dem erneuten Laden des Graphen aus.

<code-tabs name="export" />

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Der SSD300-Code von LibreYOLO wurde nicht aus der Caffe-Veröffentlichung der Autoren des Papers portiert, sondern leitet sich von der BSD-3-Clause-Implementierung SSD300 aus torchvision ab. Das oben verlinkte Repository ist daher die Upstream-Quelle. Die VGG16-Gewichte des Backbones gehen auf das vollständig konvolutionale, reduzierte VGGNet aus Oxford zurück, das Karen Simonyan und Andrew Zisserman unter CC BY 4.0 veröffentlicht haben.

</provenance-box>

## Zitieren

<citation-block />
