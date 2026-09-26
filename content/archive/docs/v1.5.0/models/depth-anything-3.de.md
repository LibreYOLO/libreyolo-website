---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: 'Depth Anything 3: monokulare Tiefe in LibreYOLO vorhersagen'
description: >-
  Depth Anything 3 in LibreYOLO für monokulare Tiefenschätzung nutzen. Den
  Checkpoint DA3MONO-LARGE installieren, vorhersagen, validieren und
  exportieren, Apache-2.0.
lead: >-
  Depth Anything 3 ist ein schlichter DINOv2-Transformer, der darauf trainiert
  ist, Tiefe und Kamerageometrie aus einer oder mehreren Ansichten
  vorherzusagen, ganz ohne architektonische Spezialisierung. LibreYOLO portiert
  seinen DA3MONO-LARGE-Checkpoint für die Aufgabe Tiefenschätzung: Vorhersage
  und Zero-Shot-Validierung, ohne Trainingspfad.
keywords:
  - Depth Anything 3
  - DA3
  - monokulare tiefenschätzung
  - tiefenschätzung python
  - tiefenkarte aus einem bild berechnen
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Tiefenkarte auslesen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnything3l-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: dicht (H, W), höher = näher

        raw = depth.data                # Tensor, ohne Einheit und ohne feste
        Skala

        normalized = depth.normalized() # auf [0, 1] skaliert zur Visualisierung
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung, ein exportiertes Artefakt
        # lädt wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## Installation

Depth Anything 3 braucht kein optionales Extra. Alles, was es importiert,
steckt in der Basis-Installation.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

`result.depth_map` trägt eine dichte relative inverse Tiefenkarte: höhere
Werte bedeuten näher an der Kamera, und die Werte haben keine metrische
Einheit und keine bildübergreifende Skala. Der Upstream-Checkpoint gibt
positive relative Tiefe aus; LibreYOLOs Netzwerk-Wrapper invertiert sie und
reproduziert die offizielle Himmelsbehandlung, damit die Ausgabe dem
gemeinsamen Tiefen-Kontrakt von LibreYOLO folgt. `save=True` schreibt eine
farbcodierte Visualisierung dieser Karte auf die Festplatte; `Results.plot()`
deckt diese Familie nicht ab, da es nur für Oberflächennormalen und Kanten
definiert ist. Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und
den Umgang mit Ergebnissen.

## Varianten

Eine Größe, `l`, bei fester Eingabeauflösung. Das Upstream-Projekt DA3
veröffentlicht außerdem Small- und Base-Any-View-Checkpoints, einen Checkpoint
für metrische Tiefe sowie Nested- und Giant-Checkpoints; LibreYOLO stellt
keinen davon bereit. Metrische Tiefe braucht einen anderen öffentlichen
Kontrakt als LibreYOLOs Aufgabe der relativen inversen Tiefe, und die
Any-View- und Nested-Checkpoints brauchen eine Mehrbild-Kamera-API, die
LibreYOLO nicht anbietet. Die Large- und Giant-Any-View-Checkpoints stehen
zudem unter CC-BY-NC-4.0 und werden von keinem LibreYOLO-Download-Pfad
referenziert.

Training wird für diese Familie nicht angeboten.
`LibreDepthAnything3.train()` löst bedingungslos `NotImplementedError` aus;
trainiere upstream und konvertiere einen kompatiblen
DA3MONO-LARGE-Checkpoint mit `weights/convert_depth_anything3_weights.py`.

## Validierung

`val()` führt den gemeinsamen Tiefen-Validator aus: er richtet jede Vorhersage
mit einer Skalierung und Verschiebung nach kleinsten Quadraten pro Bild an
ihrer Ground Truth aus und meldet dann die üblichen Zero-Shot-Metriken für
relative Tiefe, AbsRel, RMSE und die drei Delta-Schwellenwerte.

<code-tabs name="val" />

## Export

<export-matrix />

Der Export ist für diese Familie auf fünf Formate beschränkt: ONNX,
TorchScript, ExecuTorch, TensorRT und OpenVINO. Jedes andere angeforderte
Format löst `NotImplementedError` aus, statt eine unvalidierte Konvertierung
zu versuchen. Ein exportiertes Artefakt lädt über seine Dateiendung wieder
durch `LibreYOLO()`, sodass sich eine `.onnx`- oder `.engine`-Datei wie ein
Checkpoint verhält und dieselben `Results` liefert, mit `depth_map` an Stelle
von Boxen.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
