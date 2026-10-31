---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: semantische Segmentierung in LibreYOLO'
description: >-
  Nutze LingBot-Vision in LibreYOLO für semantische Segmentierung auf einem
  ViT-Backbone unter Apache-2.0. Installiere, sage vorher, trainiere, validiere
  und exportiere die Größen s/b/l.
lead: >-
  LingBot-Vision ist eine Familie selbstüberwachter Vision-Transformer-Backbones
  von Robbyant, die mit grenzzentrierter maskierter Modellierung für dichte
  räumliche Wahrnehmung trainiert wurden. LibreYOLO verbindet das Backbone mit
  einem dichten Head und unterstützt eine Aufgabe: semantische Segmentierung.
keywords:
  - lingbot-vision
  - semantische segmentierung
  - vision transformer
  - selbstüberwachtes vortraining
  - boundary modeling
  - robbyant
  - dichte vorhersage
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (lineare Probe)
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Das Backbone ist standardmäßig gemäß dem Upstream-Auswertungsprotokoll
        # eingefroren: Nur der dichte 1x1-Head wird trainiert.
        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Vollständiges Fine-Tuning
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreLingBotVisions-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Installation

LingBot-Vision benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

`result.semantic_mask` enthält die dichte Klassenkarte: `.data` ist ein Tensor
der Form `(H, W)` mit Klassen-IDs in der ursprünglichen Bildgröße. `.classes`
führt die tatsächlich vorhandenen Klassen-IDs auf. `result.boxes` ist `None`,
weil es keine instanzbezogenen Erkennungen gibt. `conf` und `iou` werden aus
Gründen der API-Parität akzeptiert, verändern die Ausgabe aber nicht. Das
Modell gibt eine Klasse pro Pixel und keine zu filternden Erkennungen zurück.
Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

Es gibt drei veröffentlichte Größen s, b und l, die aus einem ViT-g/16-Lehrer
mit 1.1 Milliarden Parametern destilliert wurden. Der Lehrer selbst, Größe `g`,
kann in LibreYOLO geladen und nachtrainiert werden. LibreYOLO hostet jedoch
keinen eigenen `g`-Checkpoint.

<checkpoint-table />

## Training

`train()` führt das Fine-Tuning eines veröffentlichten Checkpoints durch. Das
Standardrezept entspricht der linearen Probe des Upstream-Berichts: Das
ViT-Backbone ist eingefroren, und nur der dichte 1x1-Head wird trainiert. So
wurden auch die oben von LibreYOLO gehosteten Gewichte erzeugt. Übergib
`freeze_backbone=False`, um stattdessen das gesamte Netzwerk nachzutrainieren,
und senke `lr0` entsprechend.

<code-tabs name="train" />

Unter [Training](/docs/train) findest du Datensätze, Datenaugmentierung,
Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln zurück: mIoU und
Pixel-Accuracy, gemessen auf einem beliebigen Datensatz in dem Format, das du
für das Training verwendet hast.

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

<provenance-box>

Das Upstream-Release beschreibt sein ViT als auf der von Meta AI
veröffentlichten DINOv2/DINOv3-Architektur aufgebaut. Robbyant verteilt die
eigene Implementierung unter Apache-2.0. Diese LibreYOLO-Portierung wurde nur
aus dem Robbyant-Repository erstellt, nie aus dem DINOv2- oder DINOv3-Code von
Meta.

</provenance-box>

## Zitieren

<citation-block />
