---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: Bilder mit LibreSwin von LibreYOLO klassifizieren'
description: >-
  Nutze Swin-Transformer-Klassifikatoren mit LibreYOLO für Vorhersage,
  Validierung und Export. MIT-Gewichte; Fine-Tuning wird noch nicht unterstützt.
lead: >-
  Swin Transformer V1 ist ein hierarchischer Vision Transformer, der Attention
  innerhalb verschobener lokaler Fenster statt über das gesamte Bild berechnet.
  LibreYOLO stellt vier Größen für die Bildklassifizierung bereit.
keywords:
  - Swin Transformer
  - hierarchischer Vision Transformer
  - Shifted-Window-Attention
  - Bildklassifizierung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwint-cls.pt")


        # data ist ein Verzeichnis mit train/- und val/-Splits aus
        Klassenordnern

        # (ImageFolder-Layout), keine Datensatz-YAML-Datei.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreSwint-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Installation

Swin benötigt kein optionales Extra. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Ein Klassifikator gibt `result.probs` statt `result.boxes` zurück. `top1` und `top5` enthalten Klassenindizes, `top1conf` und `top5conf` die zugehörigen Konfidenzwerte. Jede Größe ist auf eine Eingabe von 224 px festgelegt, da die letzte Attention-Stufe für diese Auflösung aufgebaut ist. Vorhersage, Validierung und Export lösen einen Fehler aus, wenn du einen anderen Wert für `imgsz` übergibst. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Die vier Größen von Tiny bis Large basieren auf demselben Shifted-Window-Tower und unterscheiden sich in Embedding-Breite und Stufentiefe. Large wurde auf ImageNet-22k vortrainiert und auf ImageNet-1k feinabgestimmt. Die anderen drei wurden direkt auf ImageNet-1k trainiert. LibreYOLO stellt diese Familie nur für die Inferenz bereit. Unterstützt werden Vorhersage, Validierung im ImageNet-Stil mit Top-1 und Top-5 sowie Export. Das Upstream-Trainingsrezept für ImageNet ist nicht implementiert.

## Validierung

`val()` arbeitet mit einem Split im ImageFolder-Stil, also einem Verzeichnis mit den Unterordnern `train/` und `val/` und je einem Ordner pro Klasse. Die Methode gibt die Genauigkeit für Top-1 und Top-5 zurück.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Unter [Export](/docs/export) findest du die Argumente, die jedes Format akzeptiert, sowie zusätzliche Argumente einzelner Formate.

<code-tabs name="export" />

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
