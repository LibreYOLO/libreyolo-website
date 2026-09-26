---
title: ViT
families:
  - vit
seo_title: 'ViT: klassische Vision-Transformer-Klassifikatoren in LibreYOLO'
description: >-
  Nutze ViT-Klassifikatoren mit LibreYOLO für Vorhersage, Validierung und
  Export. Apache-2.0-AugReg-Gewichte; Fine-Tuning wird noch nicht unterstützt.
lead: >-
  Der klassische Vision Transformer ist ein reiner Transformer für Bild-Patches
  fester Größe mit einem gelernten Klassentoken und ohne Faltungen. LibreYOLO
  stellt vier mit AugReg vortrainierte Größen für die Bildklassifizierung
  bereit.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - Bildklassifizierung
  - Transformer-Klassifikator
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreViTti-cls.pt")


        # data ist ein Verzeichnis mit train/- und val/-Splits aus
        Klassenordnern

        # (ImageFolder-Layout), keine Datensatz-YAML-Datei.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreViTti-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Installation

ViT benötigt kein optionales Extra. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Ein Klassifikator gibt `result.probs` statt `result.boxes` zurück. `top1` und `top5` enthalten Klassenindizes, `top1conf` und `top5conf` die zugehörigen Konfidenzwerte. Die Vorverarbeitung skaliert das Bild und beschneidet es mittig auf eine feste Eingabe von 224 px. Dafür nutzt sie das AugReg-Auswertungsrezept von timm mit bikubischer Interpolation und einem Beschnittfaktor von 0,9. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Die vier Größen von Tiny bis Large verwenden denselben festen Graphen mit 224 px und Patchgröße 16. Sie unterscheiden sich in Embedding-Breite und Transformer-Tiefe. LibreYOLO stellt diese Familie nur für die Inferenz bereit. Unterstützt werden Vorhersage, Validierung im ImageNet-Stil mit Top-1 und Top-5 sowie Export. Das AugReg-Rezept für das Fine-Tuning ist nicht implementiert.

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
