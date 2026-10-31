---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: monokulare Tiefenschätzung in LibreYOLO'
description: "Führe relative Tiefeninferenz mit MiDaS in LibreYOLO aus. Die Checkpoints s und l verwenden LibreYOLO-Mirrors unter der MIT-Lizenz des Herausgebers."
lead: >-
  MiDaS ist eine monokulare relative Tiefenschätzung, die mit einem skalierungs-
  und verschiebungsinvarianten Loss auf gemischten Datensätzen trainiert wurde.
  Diese Arbeit etablierte das später von anderen Familien wiederverwendete
  Protokoll zur Zero-Shot-Tiefenübertragung. LibreYOLO unterstützt es für die
  Tiefenaufgabe: Vorhersage und Zero-Shot-Validierung ohne Trainingspfad.
keywords:
  - midas tiefenschätzung
  - monokulare tiefe
  - dpt
  - relative tiefe
  - depth map python
  - zero-shot tiefenschätzung
last_verified: "1.6.0"
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Lädt den gespiegelten Checkpoint bei der ersten Verwendung.

        model = LibreYOLO("LibreMiDaSl-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)


        depth = result.depth_map

        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Kleine Variante
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # EfficientNet-Lite3-Encoder, kleiner und schneller als DPT-Large der
        Größe l.

        model = LibreYOLO("LibreMiDaSs-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreMiDaSl-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 64537aa3ad3a6f8f
---

## Installation

MiDaS benötigt das Extra `midas` für seine timm-Encoder.

```bash
pip install "libreyolo[midas]"
```

## Vorhersage

Die Checkpoints s und l werden von LibreYOLO-Mirrors unter der MIT-Lizenz des Herausgebers heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

`result.depth_map` enthält eine dichte Karte der relativen inversen Tiefe: Höhere Werte bedeuten näher an der Kamera; die Werte haben weder eine metrische Einheit noch eine bildübergreifende Skala. `save=True` speichert eine farbcodierte Visualisierung der Karte; `Results.plot()` stellt die Tiefenkarte dar. Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Es gibt zwei Varianten mit unterschiedlichen Encodern, nicht nur verschiedene
Skalierungen desselben Encoders. `s` ist MiDaS v2.1 Small mit einem
EfficientNet-Lite3-Encoder. `l` ist DPT-Large mit einem ViT-L/16-Encoder und
dem von MiDaS für dichte Vorhersagen eingeführten DPT-Decoder. Auch ihre
Vorverarbeitung unterscheidet sich: `s` verwendet eine Seitenverhältnis-Skalierung
mit Obergrenze und die ImageNet-Normalisierung für Mittelwert und
Standardabweichung. `l` verwendet eine minimale Seitenverhältnis-Skalierung
mit Mittelwert und Standardabweichung 0.5. Wähle `s` für ein leichteres CNN
und `l` für die Accuracy des Transformer-Decoders.

Für diese Familie wird kein Training angeboten. `LibreMiDaS.train()` löst
immer `NotImplementedError` aus.

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

<code-tabs name="export" />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
