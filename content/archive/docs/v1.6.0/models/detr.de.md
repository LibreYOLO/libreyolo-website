---
title: DETR
families:
  - detr
seo_title: 'DETR: Vorhersage und Export unter Apache-2.0'
description: >-
  Führe DETR, den ursprünglichen Detection Transformer, in LibreYOLO aus.
  Installiere, sage vorher, validiere und exportiere vier ResNet-basierte
  Größen, alle unter Apache-2.0.
lead: >-
  DETR ist der ursprüngliche Detection Transformer. Statt Anchors oder eines
  dichten Rasters verwendet er einen Transformer-Decoder mit ungarischer
  Zuordnung, um eine feste Menge von Objekten vorherzusagen. LibreYOLO bietet
  vier Größen für die reine Erkennungsinferenz.
keywords:
  - detr objekterkennung
  - detection transformer python
  - object detection transformer
  - hungarian matching
  - transformer decoder
  - meta ai detr
  - facebook ai research detr
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() gibt ein einfaches dict zurück, kein Objekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Installation

DETR benötigt kein optionales Zusatzpaket. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. `conf` und `max_det` filtern die Auswahl der Queries. `iou` wird aus
Gründen der API-Parität akzeptiert, hat aber keine Wirkung, weil der Decoder
Mengen vorhersagt und keinen NMS-Schritt verwendet. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

DETR unterstützt in LibreYOLO nur Inferenz. Upstream wird das Modell 500
Epochen lang mit ungarischer Zuordnung trainiert. Dieses Rezept ist hier nicht
implementiert, daher löst `train()` den Fehler `NotImplementedError` aus.

## Varianten

Vier Checkpoints kombinieren zwei Backbone-Tiefen, ResNet-50 oder ResNet-101,
mit einer optional dilatierten C5-Stufe. Bei den DC5-Varianten behält die letzte
Backbone-Stufe die volle Auflösung bei, statt sie weiter zu reduzieren. Dadurch
liest der Decoder bei derselben Eingabegröße eine feinere Feature Map. Alle vier
verwenden 100 gelernte Objekt-Queries, einen sechsschichtigen
Transformer-Encoder-Decoder und dieselbe Eingabeauflösung.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück. [Export](/docs/export) führt die
Argumente auf, die von den einzelnen Formaten akzeptiert werden.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>
