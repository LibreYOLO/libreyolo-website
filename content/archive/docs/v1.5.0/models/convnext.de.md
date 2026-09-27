---
title: ConvNeXt
families:
  - convnext
seo_title: 'ConvNeXt: Training, Validierung und Export unter Apache-2.0'
description: >-
  ConvNeXt in LibreYOLO für die Bildklassifikation nutzen. LibreConvNeXt
  tiny/small/base installieren, Vorhersagen treffen, mit LoRA nachtrainieren,
  validieren und exportieren.
lead: >-
  ConvNeXt ist ein Bildklassifikator, der vollständig aus gewöhnlichen Faltungen
  aufgebaut ist, Block für Block von einem ResNet in Richtung der
  Designentscheidungen eines Vision Transformers modernisiert. LibreYOLO
  unterstützt ihn für eine Aufgabe: die Klassifikation.
keywords:
  - ConvNeXt
  - ConvNeXt tiny
  - bildklassifikation python
  - convnext modell trainieren
  - imagenet klassifikator
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über das Dateisuffix: ein exportiertes Artefakt
        # lädt wie ein Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreConvNeXtt-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 1682cc69cf2925e6
---

## Installation

ConvNeXt braucht kein optionales Extra. Alles, was es importiert, steckt in
der Basisinstallation.

```bash
pip install libreyolo
```

Fine-Tuning mit Adaptern über `lora=True` ist die Ausnahme und braucht das
Extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt ist dasselbe, das jede Familie liefert,
ein anderes Modell einzusetzen ist also eine Änderung von einer Zeile. Ein
Klassifikator führt keine Boxen und keine Masken mit: `result.probs` enthält
die Vorhersage für das ganze Bild, mit `top1`, `top5`, `top1conf` und
`top5conf`. `conf`, `iou` und `max_det` werden aus Gründen der API-Parität
akzeptiert, haben aber keine Wirkung, denn bei einem einzelnen
Wahrscheinlichkeitsvektor gibt es nichts, worauf ein Schwellenwert oder eine
Unterdrückung wirken könnte. Siehe [Vorhersage](/docs/predict) für Quellen,
Streaming und den Umgang mit Ergebnissen.

## Varianten

Drei Größen, tiny/small/base, alle gleich trainiert und gleich ausgewertet,
die Wahl ist also ein direkter Tausch von Parameterzahl gegen Accuracy. Die
Aufgabe liegt fest: jede Größe deckt nur die Klassifikation ab. Der Name der
Gewichtsdatei endet bei jeder Größe auf `-cls.pt`, und genau dieses Suffix
liest die Factory, um an diese Familie zu routen; ein `task=`-Argument ist
nicht nötig.

## Training

Das Fine-Tuning startet vom veröffentlichten ImageNet-Backbone und baut die
letzte Klassifikatorschicht automatisch auf die Klassenzahl des
Zieldatensatzes um.

<code-tabs name="train" />

Ohne weitere Angaben läuft das Training 100 Epochen mit `lr0=1e-3`, AdamW,
einem Batch von 64 und Early Stopping nach 50 Epochen ohne Verbesserung.
`data` akzeptiert ein Datensatzverzeichnis (`train/` und `val/`, ein Ordner
pro Klasse), einen bekannten Kurznamen wie `imagenette160` oder eine
`.zip`-URL. Die Blöcke von ConvNeXt enthalten die `nn.Linear`-MLPs, die LoRA
braucht, deshalb wird `lora=True` hier unterstützt und schleust Adapter in die
Block-MLPs ein, statt das ganze Backbone nachzutrainieren.

Siehe [Training](/docs/train) für Datensätze, Augmentierung, Multi-GPU und
Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln zurück. Bei der
Klassifikation sind das die Top-1- und die Top-5-Accuracy über dem
Validierungs-Split.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt lädt über sein Dateisuffix wieder durch
`LibreYOLO()`, eine `.onnx`- oder `.engine`-Datei verhält sich also wie ein
Checkpoint und liefert dasselbe `Results`. [Export](/docs/export) listet die
Argumente auf, die jedes Format akzeptiert, und die zusätzlichen, die einige
davon mitbringen.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

In dieser Familie wird nur ConvNeXt V1 ausgeliefert. Die kleinen
vortrainierten Checkpoints von ConvNeXt-V2 stehen unter CC-BY-NC 4.0 und sind
bewusst ausgeschlossen, denn nicht-kommerzielle Gewichte lassen sich nicht
innerhalb einer MIT-lizenzierten, kommerziell nutzbaren Bibliothek
weiterverbreiten.

</provenance-box>

## Zitieren

<citation-block />
