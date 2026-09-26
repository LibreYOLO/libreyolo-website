---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: Objekterkennung in LibreYOLO'
description: >-
  Führe CenterNet (Objects as Points) in LibreYOLO mit den Backbones ResDCN-18
  und DLA-34 aus. Vorhersage, Validierung und Export nach ONNX unter MIT. Kein
  Trainingspfad.
lead: >-
  CenterNet modelliert ein Objekt als Mittelpunkt seiner Bounding Box und
  regressiert jede weitere Eigenschaft aus einem Heatmap-Peak, braucht also
  keine Anchors und keinen Non-Maximum-Suppression-Schritt. LibreYOLO liefert es
  als reinen Inferenz-Detektor aus.
keywords:
  - centernet objekterkennung
  - objects as points
  - ankerfreier detektor
  - heatmap objekterkennung
  - centernet nach onnx exportieren
  - centernet in python nutzen
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # ONNX-Export braucht Opset 16 oder neuer: die Upsampling-Stufe mit
        # Deformable Convolution wird zu GridSample, neu in Opset 16.
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Die Factory routet über die Dateiendung, ein exportiertes Artefakt
        # lädt wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Installation

CenterNet braucht kein optionales Extra. Alles, was es importiert, steckt in
der Basisinstallation.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt ist dasselbe, das jede Familie zurückgibt,
ein anderer Detektor ist also eine Änderung von einer Zeile. `conf` und
`max_det` filtern die nach Rang sortierten Heatmap-Peaks; `iou` wird aus
Gründen der API-Parität akzeptiert, hat aber keine Wirkung, weil das
Top-k-Peak-Decoding von CenterNet keinen Schritt zur Box-IoU-Unterdrückung
braucht. Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und die
Verarbeitung der Ergebnisse.

## Varianten

Zwei Backbones. `resdcn18` kombiniert einen ResNet-18-Trunk mit Upsampling per
Deformable Convolution; `dla34` kombiniert einen DLA-34-Trunk mit iterativem
Deep-Aggregation-Upsampling. Beide speisen dieselben drei dichten Heads
(Heatmap, Breite/Höhe, Offset) und arbeiten auf derselben Eingabefläche.

## Validierung

`val()` liefert ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95, gemessen an jedem Datensatz in dem Format, auf dem du
trainiert hast.

<code-tabs name="val" />

## Export

<export-matrix />

Der ONNX-Export verlangt Opset 16 oder neuer: die Upsampling-Stufe mit
Deformable Convolution wird in beiden Backbones auf den ONNX-Operator
`GridSample` abgebildet, den Opset 16 eingeführt hat. Forderst du ein älteres
Opset an, bricht der Export ab, bevor das Tracing beginnt.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Der ResDCN-18-Graph nennt zusätzlich Microsofts MIT-lizenziertes
human-pose-estimation.pytorch, und der DLA-34-Graph nennt Fisher Yus
DLA-Implementierung unter BSD-3-Clause. LibreYOLO liefert die ursprüngliche
DCNv2-Erweiterung des Upstream-Projekts nicht mit; die native Ausführung nutzt
stattdessen `deform_conv2d` aus torchvision unter BSD-3-Clause, und die
portable Implementierung nur für den Export wurde eigens für LibreYOLO
geschrieben.

</provenance-box>

## Zitieren

<citation-block />
