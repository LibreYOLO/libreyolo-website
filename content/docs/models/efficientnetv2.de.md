---
title: EfficientNetV2
families:
  - efficientnetv2
seo_title: 'EfficientNetV2: Training, Validierung und Export unter Apache-2.0'
description: >-
  Nutze EfficientNetV2 in LibreYOLO für die Bildklassifizierung. Installiere,
  sage vorher, führe Fine-Tuning durch, validiere und exportiere
  LibreEfficientNetV2 b0 bis b3.
lead: >-
  EfficientNetV2 ist ein Bildklassifikator, dessen Tiefe, Breite und blockweise
  Auswahl durch Neural Architecture Search ermittelt wurden. Dabei wurden
  Accuracy und Trainingsgeschwindigkeit gemeinsam optimiert und nicht nur die
  Accuracy. LibreYOLO unterstützt es für eine Aufgabe: Klassifizierung.
keywords:
  - efficientnetv2 bildklassifizierung
  - efficientnetv2 b0 python
  - image classification
  - neural architecture search
  - mbconv
  - imagenet klassifikator
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientNetV2b0-cls.pt source=cat.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160
        epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientNetV2b0-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=onnx

        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=tensorrt
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreEfficientNetV2b0-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: ad3ff140aad824bd
---

## Installation

EfficientNetV2 benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Modell erfordert daher nur eine Änderung in einer
Zeile. Ein Klassifikator enthält weder Boxen noch Masken: `result.probs`
enthält die Ganzbildvorhersage mit `top1`, `top5`, `top1conf` und `top5conf`.
`conf`, `iou` und `max_det` werden aus Gründen der API-Parität akzeptiert,
haben aber keine Wirkung. Bei einem einzelnen Wahrscheinlichkeitsvektor gibt
es nichts, worauf ein Schwellenwert oder eine Unterdrückung angewendet werden
könnte. Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

Es gibt vier Größen von b0 bis b3. Jede wird mit ihrer eigenen Auflösung und
ihrem eigenen Crop-Verhältnis ausgewertet, statt eine Eingabegröße für die
gesamte Familie zu verwenden. Die Wahl einer Größe ist ein direkter Tausch von
Parameteranzahl gegen Accuracy. Die Aufgabe ist fest: Jede Größe unterstützt
nur Klassifizierung. Der Dateiname der Gewichte endet bei jeder Größe mit
`-cls.pt`. An diesem Suffix erkennt die Factory die Familie. Ein Argument
`task=` ist nicht erforderlich.

## Training

Das Fine-Tuning beginnt mit dem veröffentlichten ImageNet-Backbone und baut die
letzte Klassifikatorschicht automatisch für die Klassenanzahl des Zieldatensatzes
neu auf. Wenn du `imgsz` nicht ausdrücklich festlegst, wird standardmäßig die
eigene Auswertungsauflösung der Größe verwendet.

<code-tabs name="train" />

Ohne Änderungen läuft der Trainer 100 Epochen mit `lr0=1e-3`, AdamW, einem
Batch von 64 und Early Stopping nach 50 Epochen ohne Verbesserung. `data`
akzeptiert ein Datensatz-Stammverzeichnis (`train/` und `val/` mit einem Ordner
pro Klasse), einen bekannten Kurznamen wie `imagenette160` oder eine `.zip`-URL.
`lora=True` wird hier nicht unterstützt und löst einen Fehler aus. LoRA richtet
sich in LibreYOLO an Transformer-Komponenten mit `nn.Linear`-Schichten, die in
den MBConv-Blöcken dieser Familie nicht vorkommen.

Unter [Training](/docs/train) findest du Datensätze, Datenaugmentierung,
Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln zurück. Bei der
Klassifizierung sind dies Top-1- und Top-5-Accuracy auf dem Validierungssplit.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück. [Export](/docs/export) führt die
Argumente auf, die jedes Format akzeptiert, sowie die Zusatzpakete, die einige
davon benötigen.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>
