---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: Vorhersage und Export unter Apache-2.0'
description: >-
  Führe LW-DETR in LibreYOLO für Echtzeit-Objekterkennung aus. Installiere, sage
  vorher, validiere und exportiere fünf ViT-basierte Größen, alle unter
  Apache-2.0.
lead: >-
  Ein Detection Transformer mit einfachem ViT, den Baidu als Echtzeitalternative
  zu YOLO-Detektoren positioniert hat. LibreYOLO bietet fünf Größen für die
  reine Erkennungsinferenz.
keywords:
  - lw-detr
  - detection transformer
  - echtzeit objekterkennung
  - plain vit
  - detr
  - baidu
  - atten4vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() gibt ein einfaches dict zurück, kein Objekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreLWDETRt.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Installation

LW-DETR benötigt kein optionales Zusatzpaket. Alle Importe sind in der Basisinstallation enthalten.

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

LW-DETR unterstützt in LibreYOLO nur Inferenz. Upstream wird das Modell mit
One-to-Many-Überwachung von Group-DETR über mehrere Query-Gruppen und einem
IoU-bewussten Klassifizierungs-Loss trainiert. Dieses Rezept ist hier nicht
angeschlossen, daher löst `train()` den Fehler `NotImplementedError` aus.

## Varianten

Es gibt fünf Größen. Alle verwenden denselben einfachen ViT-Encoder,
mehrskaligen Projektor und Deformable-DETR-Decoder sowie dieselbe
Eingabeauflösung. Die beiden kleinsten haben dieselbe Encoder-Breite und
unterscheiden sich in der Blocktiefe. Die nächsten beiden verwenden denselben
breiteren Encoder und unterscheiden sich in der Anzahl der Projektorebenen,
die den Decoder speisen. Das größte Modell verwendet den breitesten Encoder.

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

## Zitieren

<citation-block />
