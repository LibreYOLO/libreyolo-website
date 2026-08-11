---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: leichtgewichtige monokulare Tiefenschätzung in LibreYOLO'
description: >-
  Verwende ZipDepth in LibreYOLO für die leichtgewichtige monokulare
  Tiefenschätzung. Installiere, nutze, validiere und exportiere zwei
  MIT-lizenzierte Checkpoints.
lead: >-
  ZipDepth ist ein kompaktes, reparametrisierbares CNN, das aus Depth Anything
  V2 Large destilliert wurde und eine dichte relative inverse Tiefenkarte
  vorhersagt. LibreYOLO unterstützt es für die Tiefenaufgabe mit Vorhersage und
  Zero-Shot-Validierung, aber ohne Trainingspfad.
keywords:
  - ZipDepth
  - monokulare Tiefenschätzung
  - Tiefenmodell für Edge-Geräte
  - relative Tiefe
  - Tiefenkarte
  - reparametrisierbares CNN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: NPU-/Edge-Checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Derselbe Encoder mit einem Upsampling-Kopf ohne Unfold für Compiler
        ohne

        # Gather-/Unfold-Unterstützung. Die Ausgabe entspricht visuell dem
        b-Checkpoint.

        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreZipDepthb-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Installation

ZipDepth benötigt kein optionales Extra. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

`result.depth_map` enthält eine dichte relative inverse Tiefenkarte. Höhere Werte bedeuten eine geringere Entfernung zur Kamera. Die Werte besitzen weder eine metrische Einheit noch eine bildübergreifende Skala. `save=True` schreibt eine farbcodierte Visualisierung dieser Karte auf den Datenträger. `Results.plot()` unterstützt diese Familie nicht, da die Methode nur für Oberflächennormalen und Kanten definiert ist. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Die beiden Checkpoints besitzen dieselbe Encoder-Kapazität und unterscheiden sich nur im trainierten Upsampling-Kopf. `b` verwendet konvexes Upsampling und läuft auf GPU oder CPU. `bnpu` ersetzt dies durch einen Decoder ohne Unfold für NPU- und Edge-Compiler, die Gather oder Unfold nicht unterstützen. Seine Ausgabe wird als visuell gleichwertig zu `b` beschrieben. Wähle `bnpu` für eine eingeschränkte Ziellaufzeitumgebung, andernfalls `b`.

Beide Checkpoints wurden aus Pseudo-Labels von Depth Anything V2 Large destilliert. Diese Familie bildet damit die kompakte, Edge-orientierte Stufe der Tiefenaufgabe von LibreYOLO neben den größeren Encodern von Depth Anything V2.

Für diese Familie wird kein Training angeboten. `LibreZipDepth.train()` löst bedingungslos `NotImplementedError` aus. Das Upstream-Rezept destilliert Pseudo-Labels über eine große Bildmenge und lässt sich nicht als LibreYOLO-Trainingslauf reproduzieren. Trainiere im Upstream-Projekt unter [fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth) und konvertiere das Ergebnis mit `weights/convert_zipdepth_weights.py`.

## Validierung

`val()` führt den gemeinsamen Tiefenvalidator aus. Er richtet jede Vorhersage anhand von Skalierung und Verschiebung nach der Methode der kleinsten Quadrate bildweise an der Ground Truth aus und meldet anschließend die üblichen Zero-Shot-Metriken für relative Tiefe: AbsRel, RMSE und die drei Delta-Schwellenwerte.

<code-tabs name="val" />

## Export

<export-matrix />

Der Export verwendet einen dichten Vertrag mit fester Auflösung. Das Quellbild wird gestreckt auf den exportierten Canvas skaliert und die zurückgegebene Tiefenkarte anschließend wieder auf den ursprünglichen Canvas gebracht. Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.ncnn`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück, wobei `depth_map` an die Stelle von Boxen tritt.

<code-tabs name="export" />

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
