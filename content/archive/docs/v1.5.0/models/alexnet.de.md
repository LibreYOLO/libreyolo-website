---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: den klassischen ImageNet-Klassifikator in LibreYOLO ausführen'
description: >-
  AlexNet mit LibreYOLO vorhersagen, validieren und exportieren.
  torchvision-Gewichte unter BSD-3-Clause; Fine-Tuning wird noch nicht
  unterstützt.
lead: >-
  AlexNet ist das Convolutional Network, das die ILSVRC 2012 gewonnen und die
  Deep-Learning-Ära in der Computer Vision mit angestoßen hat. LibreYOLO liefert
  die spätere Single-Tower-Revision der Architektur für die Bildklassifikation.
keywords:
  - AlexNet
  - ImageNet
  - bildklassifizierung python
  - convolutional neural network erklärt
  - vortrainierter bildklassifikator
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data ist ein Wurzelverzeichnis mit train/ und val/ als Klassenordner
        # (ImageFolder-Layout), kein Dataset-YAML.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory entscheidet anhand der Dateiendung, ein exportiertes
        # Artefakt lädt also wie jeder Checkpoint und liefert dasselbe Results.
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Installation

AlexNet braucht kein optionales Extra. Alles, was es importiert, steckt in der
Basis-Installation.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

Ein Klassifikator liefert `result.probs` statt `result.boxes`: `top1` und
`top5` geben die Klassenindizes, `top1conf` und `top5conf` deren Confidence.
Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und die
Weiterverarbeitung der Ergebnisse.

## Varianten

Eine Größe. Der ausgelieferte Graph ist die spätere Single-Tower-Revision, die
torchvision veröffentlicht hat, mit 64 Filtern in der ersten Schicht und ohne
Local Response Normalization, nicht die ursprüngliche Zwei-GPU-Architektur von
2012. LibreYOLO liefert diese Familie nur für die Inferenz: Vorhersage,
Validierung im ImageNet-Stil mit Top-1/Top-5 und Export werden unterstützt,
Fine-Tuning ist nicht implementiert.

## Validierung

`val()` läuft gegen einen Split im ImageFolder-Stil (ein Verzeichnis mit den
Unterordnern `train/` und `val/`, ein Ordner pro Klasse) und liefert die Top-1-
und Top-5-Accuracy.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt lädt über `LibreYOLO()` anhand seiner Dateiendung
wieder zurück, eine `.onnx`- oder `.engine`-Datei verhält sich also wie ein
Checkpoint und liefert dasselbe `Results`. [Export](/docs/export) listet die
Argumente auf, die jedes Format akzeptiert, und die Extras, die einige davon
ergänzen.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>
