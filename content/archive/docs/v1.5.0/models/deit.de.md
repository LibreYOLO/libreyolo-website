---
title: DeiT
families:
  - deit
seo_title: 'DeiT-Bildklassifikator: Vorhersage, Validierung, Export'
description: >-
  DeiT-Bildklassifikatoren in LibreYOLO ausführen: eine eingefrorene
  Museumsfamilie nur für die Inferenz, in den Größen tiny, small und base, unter
  Apache-2.0.
lead: >-
  DeiT (Data-efficient image Transformer) ist ein schlichter
  Vision-Transformer-Klassifikator, der allein auf ImageNet-1k trainiert wurde,
  ohne zusätzliche Vortrainingsdaten. LibreYOLO führt die Patch-16-Größen tiny,
  small und base als eingefrorenes Ausstellungsstück, nur für die Inferenz.
keywords:
  - DeiT
  - Vision Transformer
  - ViT
  - bildklassifizierung python
  - vision transformer erklärt
  - imagenet klassifikator python
  - vortrainierter bildklassifikator
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory entscheidet anhand der Dateiendung, ein exportiertes
        # Artefakt lädt also wie jeder Checkpoint und liefert dasselbe Results.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## Installation

DeiT braucht nichts über das Basispaket hinaus.

```bash
pip install libreyolo
```

## Vorhersage

Diese Familie ist reine Inferenz: `train()` löst `NotImplementedError` aus,
deshalb hat diese Seite keinen Abschnitt zum Training. Vorhersage, Validierung
und Export werden alle unterstützt. Die Gewichte werden beim ersten Aufruf von
Hugging Face geladen und lokal zwischengespeichert. Das Suffix `-cls` im
Dateinamen ist Pflicht und wählt die Klassifikationsaufgabe aus.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt trägt einen `probs`-Tensor statt `boxes`;
`top1` und `top5` indizieren die 1000 Klassen von ImageNet-1k, und `top1conf`
ist der Softmax-Score der besten Vorhersage. Jede Größe hat eine feste
Eingabeauflösung, die aus ihrem Positional Embedding folgt: die Vorverarbeitung
skaliert darauf und schneidet mittig zu, und ein abweichendes `imgsz` löst
einen Fehler aus, statt still neu abzutasten. Siehe
[Vorhersage](/docs/predict) für Quellen, Streaming und die Weiterverarbeitung
der Ergebnisse.

## Validierung

`val()` liefert ein Dictionary mit Top-1- und Top-5-Accuracy, gemessen gegen
einen Datensatz in der üblichen Ordnerstruktur aus `train/<class>/` und
`val/<class>/`.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt lädt über `LibreYOLO()` anhand seiner Dateiendung
wieder zurück, eine `.onnx`- oder `.engine`-Datei verhält sich also wie ein
Checkpoint und liefert dasselbe `Results`. Den Graphen in einer nackten Runtime
auszuführen, ganz ohne installiertes LibreYOLO, wird ebenfalls unterstützt,
dann musst du Vor- und Nachverarbeitung aber selbst schreiben.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
