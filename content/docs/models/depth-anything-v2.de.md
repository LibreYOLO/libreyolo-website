---
title: Depth Anything V2
families:
  - depth_anything
seo_title: 'Depth Anything V2: monokulare Tiefe vorhersagen und validieren'
description: >-
  Nutze Depth Anything V2 in LibreYOLO für die monokulare Tiefenschätzung.
  Installiere, sage vorher und validiere. Small nutzt Apache-2.0, Base und Large
  CC-BY-NC-4.0.
lead: >-
  Depth Anything V2 kombiniert einen DINOv2-Encoder mit einem DPT-Decoder und
  sagt aus einem einzelnen Bild eine dichte Karte der relativen inversen Tiefe
  vorher. LibreYOLO unterstützt das Modell für die Tiefenaufgabe: Vorhersage und
  Zero-Shot-Validierung ohne Trainingspfad.
keywords:
  - depth anything v2 tiefenschätzung
  - monokulare tiefe vorhersagen
  - dpt tiefenkarte
  - dinov2 depth
  - relative tiefe
  - depth map python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Tiefenkarte lesen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: dicht (H, W), größer = näher

        raw = depth.data                # Tensor ohne metrische Einheit oder
        bildübergreifende Skala

        normalized = depth.normalized() # für die Visualisierung auf [0, 1]
        skaliert
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## Installation

Depth Anything V2 benötigt kein optionales Zusatzpaket. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

`result.depth_map` enthält eine dichte Karte der relativen inversen Tiefe: Höhere
Werte bedeuten eine geringere Entfernung zur Kamera. Die Werte haben weder eine
metrische Einheit noch eine bildübergreifende Skala. `save=True` schreibt eine
farbkodierte Visualisierung dieser Karte auf den Datenträger. `Results.plot()`
unterstützt diese Familie nicht, weil es nur für Oberflächennormalen und Kanten
definiert ist. Die Eingabeauflösung muss ohne Rest durch 14 teilbar sein. Dies
entspricht dem DINOv2-Patch-Raster, auf dem der DPT-Head aufbaut. LibreYOLO prüft
das vor der Ausführung und löst andernfalls einen Fehler aus. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt vier Encoder-Größen s/b/l/g, die ViT-S/B/L/G entsprechen. Die folgende
Checkpoint-Tabelle führt nur s, b und l auf, weil kein Giant-Checkpoint
veröffentlicht ist. Alle vier verwenden dieselbe Eingabeauflösung. Mit der Wahl
einer Größe änderst du daher die Kapazität des Encoders, nicht die Bildgröße.
Auch die Lizenz spielt eine Rolle: Der Small-Checkpoint steht unter Apache-2.0,
Base und Large dagegen unter CC-BY-NC-4.0. Weitere Informationen findest du
unten unter Lizenzierung.

Für diese Familie werden weder Training noch Fine-Tuning angeboten.
`LibreDepthAnythingV2.train()` löst immer `NotImplementedError` aus. Konvertiere
stattdessen einen kompatiblen Upstream-Checkpoint mit
`weights/convert_depth_anything_v2_weights.py`.

## Validierung

`val()` führt den gemeinsamen Tiefenvalidator aus: Er richtet jede Vorhersage
mit einer bildweisen Least-Squares-Skalierung und -Verschiebung an ihrer Ground
Truth aus und meldet anschließend die üblichen relativen Zero-Shot-Tiefenmetriken
AbsRel, RMSE und die drei Delta-Schwellenwerte.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint,
gibt dasselbe `Results`-Objekt zurück und enthält `depth_map` anstelle von Boxen.
[Export](/docs/export) führt die Argumente auf, die von den einzelnen Formaten
akzeptiert werden.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
