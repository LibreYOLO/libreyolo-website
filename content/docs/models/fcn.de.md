---
title: FCN
families:
  - fcn
seo_title: 'FCN: Vorhersage und Export eines ResNet-FCN unter BSD-3-Clause'
description: >-
  Nutze FCN in LibreYOLO für die semantische Segmentierung. Installiere, sage
  vorher, validiere und exportiere die FCN-Checkpoints mit dilatiertem ResNet
  von torchvision.
lead: >-
  Ein dichter pixelweiser Klassifikator, der die vollständig verbundenen
  Schichten eines Detektors durch Faltungen ersetzt. Dadurch gibt er statt Boxen
  eine Klassenkarte mit voller Auflösung aus. LibreYOLO bietet ihn
  ausschließlich für die semantische Segmentierung an.
keywords:
  - fcn segmentierung
  - fully convolutional network
  - semantische segmentierung
  - dichte vorhersage
  - resnet fcn
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) Klassen-IDs
        print(mask.classes)      # sortierte Klassen-IDs im Bild
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreFCNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Installation

FCN benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Die semantische Segmentierung gibt eine Klassen-ID pro Pixel und keine Boxen
zurück. `result.semantic_mask` enthält daher ein Array der Form `(H, W)` in
`.data` und die Liste der im Bild vorkommenden Klassen-IDs in `.classes`.
`conf`, `iou` und `max_det` werden aus Gründen der API-Parität akzeptiert,
haben aber keine Wirkung: Das Modell weist jedem Pixel per Argmax eine Klasse
zu, ohne Confidence-Schwellenwert oder NMS-Schritt. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt zwei ResNet-Tiefen, beide mit einer festen Eingabe von 520 px. Der
Inferenzgraph der Bibliothek ist das FCN mit dilatiertem ResNet von torchvision,
nicht das VGG-basierte FCN-8s-Netzwerk mit Skip-Verbindungen aus der ursprünglichen
Veröffentlichung.

LibreYOLO trainiert FCN nicht: `train()` löst für diese Familie den Fehler
`NotImplementedError` aus. Der oben verlinkte [Support-Tier](/docs/models)
kennzeichnet sie daher als reine Inferenzfamilie. Die beiden veröffentlichten
Checkpoints sind die eigenen, mit COCO trainierten Gewichte von torchvision,
die für den LibreYOLO-Loader konvertiert wurden.

## Validierung

`val()` gibt `metrics/mIoU` und `metrics/pixel_accuracy` zurück. Diese werden
auf einem beliebigen Datensatz in dem Format gemessen, das du für das Training
verwendet hast.

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
