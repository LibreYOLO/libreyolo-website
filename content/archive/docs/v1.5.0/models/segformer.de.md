---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: semantische Segmentierung in LibreYOLO'
description: >-
  Nutze SegFormer in LibreYOLO für semantische ADE20K-Segmentierung in den
  Größen b0-b5. Installiere, sage vorher, trainiere und exportiere.
  Vortrainierte Gewichte sind nicht kommerziell nutzbar.
lead: >-
  SegFormer ist ein Transformer für semantische Segmentierung, der einen
  hierarchischen Mix-Transformer-Encoder (MiT) mit einem leichten reinen
  MLP-Decode-Head kombiniert. Er vermeidet die schweren Decoder und festen
  Positionskodierungen früherer Segmentierungs-Transformer. LibreYOLO
  unterstützt eine Aufgabe, semantische Segmentierung, in sechs Größen.
keywords:
  - segformer
  - semantische segmentierung
  - mix transformer
  - mit
  - transformer segmentierung
  - ade20k
  - dichte vorhersage
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (Fine-Tuning)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Von Grund auf neu
      language: python
      code: >
        from libreyolo.models.segformer.model import LibreSegformer


        # Ohne model_path: zufällige Initialisierung, kein Download. Der einzige
        Weg

        # zu Gewichten ohne die nicht kommerzielle Bedingung der vortrainierten
        Checkpoints.

        model = LibreSegformer(size="b0", nb_classes=150)

        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Installation

SegFormer benötigt kein optionales Zusatzpaket. Alle Importe sind in der Basisinstallation enthalten.

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
Modell gibt eine Klasse pro Pixel und keine zu filternden oder zu
deduplizierenden instanzbezogenen Erkennungen zurück. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt sechs Größen von b0 bis b5. Bei jeder Stufe wird der Mix-Transformer-Encoder
breiter und tiefer, während derselbe reine MLP-Decode-Head verwendet wird.

<checkpoint-table />

## Training

`train()` führt standardmäßig das Fine-Tuning eines veröffentlichten
Checkpoints durch. Übergib `LibreSegformer(...)` stattdessen keinen
`model_path`. Dann wird das Modell mit zufällig initialisiertem Encoder und
Head aufgebaut und von Grund auf neu trainiert. Nur auf diesem Weg entstehen
Gewichte ohne die nicht kommerzielle Beschränkung der vortrainierten
Checkpoints (siehe [Lizenzierung](#licensing)).

<code-tabs name="train" />

Ohne Änderungen folgt der Trainer dem ADE20K-Rezept der
SegFormer-Veröffentlichung: AdamW mit einer Basislernrate für das Backbone und
einer zehnmal höheren Rate für den Decode-Head, Weight Decay überall außer
LayerNorm und der Positionsfaltung im Mix-FFN sowie ein linearer Abfallplan mit
Warmup. Die vollständige Konvergenz der größeren Größen b3 bis b5 wurde nicht
validiert.

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

Encoder und Decode-Head von LibreSegformer sind eine PyTorch-Portierung der
Apache-2.0-SegFormer-Implementierung aus Hugging Face Transformers und nicht
von NVlabs/SegFormer. Das ursprüngliche NVIDIA-Repository wurde nie gelesen
oder kopiert und wird hier nur zur Zuordnung der Autoren der Veröffentlichung
genannt. Nur die vortrainierten Checkpoints oben unterliegen NVIDIAs
nicht kommerzieller Beschränkung. Architektur und eigener LibreYOLO-Code
bleiben vollständig MIT-lizenziert.

</provenance-box>

## Zitieren

<citation-block />
