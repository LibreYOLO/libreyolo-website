---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: Vorhersage und Export, Apache-2.0'
description: >-
  Deformable DETR in LibreYOLO für die Objekterkennung nutzen. Installation,
  Vorhersage, Validierung und Export von fünf Größen mit Sparse Attention, alle
  unter Apache-2.0.
lead: >-
  Deformable DETR ersetzt die dichte Cross-Attention von DETR durch spärliches,
  mehrskaliges Sampling rund um jeden Referenzpunkt, und genau das hat
  Transformer-Detektoren praktisch trainierbar gemacht. LibreYOLO liefert fünf
  Größen für die Erkennung, nur für die Inferenz.
keywords:
  - Deformable DETR
  - detr objekterkennung
  - transformer detektor python
  - sparse attention erklärt
  - objekterkennung python
  - deformable detr nach onnx exportieren
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() liefert ein einfaches dict, kein Objekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory entscheidet anhand der Dateiendung, ein exportiertes
        # Artefakt lädt also wie jeder Checkpoint und liefert dasselbe Results.
        model = LibreYOLO("LibreDeformableDETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## Installation

Deformable DETR braucht kein optionales Extra. Alles, was es importiert, steckt
in der Basis-Installation, mit einem reinen PyTorch-Kern für die mehrskalige
Deformable Attention.

```bash
pip install libreyolo
```

`libreyolo[hub-kernels]` zu installieren ist optional. Sobald das Paket
`kernels` vorhanden ist, holt LibreYOLO zur Laufzeit einen kompilierten Kernel
für die mehrskalige Deformable Attention vom Hugging Face Hub und nutzt ihn
anstelle des reinen PyTorch-Kerns; `LIBREYOLO_HUB_KERNELS=0` schaltet das
wieder ab.

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt ist dasselbe, das jede Familie liefert, der
Wechsel zu einem anderen Detektor ist also eine Änderung von einer Zeile.
`conf` und `max_det` filtern die Auswahl der Queries; `iou` wird aus Gründen
der API-Parität akzeptiert, hat aber keine Wirkung, weil der Decoder ein
Set-Predictor ohne NMS-Schritt ist. Siehe [Vorhersage](/docs/predict) für
Quellen, Streaming und die Weiterverarbeitung der Ergebnisse.

Deformable DETR ist in LibreYOLO reine Inferenz. Upstream wird mit Hungarian
Matching und einem Focal-Loss für die Klassifikation trainiert; dieses Rezept
ist hier nicht implementiert, deshalb löst `train()` einen
`NotImplementedError` aus.

## Varianten

Fünf Checkpoints decken die veröffentlichten Konfigurationen ab, alle bei
derselben Eingabeauflösung. `r50ss` beschränkt die Attention auf eine einzige
Feature-Skala; `r50ssdc5` ergänzt darüber hinaus eine dilatierte C5-Stufe im
Backbone. `r50` ist die mehrskalige Standardkonfiguration und samplet über vier
Feature-Map-Ebenen. `r50refine` ergänzt eine iterative Verfeinerung der
Bounding Boxes über die Decoder-Schichten hinweg, und `r50twostage` erzeugt
seine ersten Regionsvorschläge aus der Encoder-Ausgabe statt aus gelernten
Queries.

## Validierung

`val()` liefert ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95, gemessen an jedem Datensatz in dem Format, auf dem du
trainiert hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt lädt über `LibreYOLO()` anhand seiner Dateiendung
wieder zurück, eine `.onnx`- oder `.engine`-Datei verhält sich also wie ein
Checkpoint und liefert dasselbe `Results`. [Export](/docs/export) listet die
Argumente auf, die jedes Format akzeptiert.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
