---
title: PIDNet
families:
  - pidnet
seo_title: 'PIDNet: Echtzeitsegmentierung unter MIT vorhersagen und exportieren'
description: >-
  Nutze PIDNet in LibreYOLO für semantische Segmentierung in Echtzeit.
  Installiere, sage vorher, validiere und exportiere die Cityscapes-Checkpoints
  s/m/l unter MIT.
lead: >-
  Ein dreizweigiges Netzwerk für semantische Segmentierung, das einem von
  Proportional-, Integral- und Differentialanteilen inspirierten Design einen
  eigenen Grenzzweig hinzufügt und auf Echtzeitinferenz zielt. LibreYOLO bietet
  es ausschließlich für semantische Segmentierung an.
keywords:
  - pidnet
  - semantische segmentierung echtzeit
  - grenzbewusste segmentierung
  - cityscapes
  - dichte vorhersage
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) Klassen-IDs
        print(mask.classes)      # sortierte Klassen-IDs im Bild
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibrePIDNets-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Installation

PIDNet benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert. Das Dateinamenssuffix `-sem` ist für diese
Familie erforderlich.

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

Es gibt drei Größen, alle mit einer festen Eingabe von 1024 px. Die
veröffentlichten Checkpoints sind Konvertierungen der offiziellen, auf
Cityscapes mit 19 Klassen trainierten PIDNet-Gewichte.

LibreYOLO trainiert PIDNet nicht: `train()` löst für diese Familie den Fehler
`NotImplementedError` aus. Der oben verlinkte [Support-Tier](/docs/models)
kennzeichnet sie daher als reine Inferenzfamilie.

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

## Zitieren

<citation-block />
