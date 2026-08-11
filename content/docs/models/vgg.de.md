---
title: VGG
families:
  - vgg
seo_title: 'VGG: VGG-16/19-Bildklassifikatoren in LibreYOLO ausführen'
description: >-
  Nutze VGG-Klassifikatoren mit LibreYOLO für Vorhersage, Validierung und
  Export. BSD-3-Clause-Gewichte von torchvision; Fine-Tuning wird noch nicht
  unterstützt.
lead: >-
  VGG ist ein konvolutionaler Bildklassifikator, der auf gleichförmigen Stapeln
  kleiner 3x3-Faltungen statt größerer Filter basiert. LibreYOLO stellt die
  Varianten mit 16 und 19 Schichten jeweils ohne und mit Batch-Normalisierung
  für die Bildklassifizierung bereit.
keywords:
  - VGG
  - VGG-16
  - VGG-19
  - konvolutionales neuronales Netzwerk
  - Bildklassifizierung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreVGG16-cls.pt")


        # data ist ein Verzeichnis mit train/- und val/-Splits aus
        Klassenordnern

        # (ImageFolder-Layout), keine Datensatz-YAML-Datei.

        metrics = model.val(data="imagenet-1k/")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreVGG16-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## Installation

VGG benötigt kein optionales Extra. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Ein Klassifikator gibt `result.probs` statt `result.boxes` zurück. `top1` und `top5` enthalten Klassenindizes, `top1conf` und `top5conf` die zugehörigen Konfidenzwerte. Die Vorhersage verwendet eine feste Eingabegröße von 224 px und löst einen Fehler aus, wenn du einen anderen Wert für `imgsz` übergibst. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Es gibt vier Größen: 16 und 19 konvolutionale Schichten, jeweils als einfache und als Batch-normalisierte Variante. Die bereitgestellten Gewichte stammen aus dem späteren, von Grund auf neu ausgeführten ImageNet-Training von torchvision. Es handelt sich nicht um Konvertierungen der ursprünglichen Caffe-Veröffentlichung der Oxford-Gruppe von 2014. LibreYOLO stellt diese Familie nur für die Inferenz bereit. Unterstützt werden Vorhersage, Validierung im ImageNet-Stil mit Top-1 und Top-5 sowie Export. Fine-Tuning ist nicht implementiert.

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
