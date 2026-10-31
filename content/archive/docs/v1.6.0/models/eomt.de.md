---
title: EoMT
families:
  - eomt
seo_title: 'EoMT: semantische, Instanz- und panoptische Segmentierung vorhersagen'
description: >-
  Nutze EoMT in LibreYOLO für semantische, Instanz- und panoptische
  Segmentierung auf einem einfachen DINOv2 Vision Transformer, ohne Decoder.
  MIT-lizenziert.
lead: >-
  Ein Segmentierungsnetzwerk auf einem einfachen Vision Transformer ohne eigenen
  Pixel-Decoder: Zusätzliche gelernte Queries im Encoder selbst sagen die Masken
  vorher. LibreYOLO unterstützt es für semantische Segmentierung,
  Instanzsegmentierung und panoptische Segmentierung.
keywords:
  - eomt
  - encoder-only mask transformer
  - dinov2 segmentierung
  - panoptische segmentierung
  - instanzsegmentierung
  - semantische segmentierung
last_verified: 1.5.0
snippets:
  predict:
    - label: Semantisch
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) Klassen-IDs
        print(mask.classes)      # sortierte Klassen-IDs im Bild
    - label: Instanzsegmentierung
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -seg im Dateinamen wählt die Instanzaufgabe aus,
        # daher ist hier kein task-Argument erforderlich.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Panoptisch
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) Segment-IDs
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Semantisch
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Instanzsegmentierung
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # Masken
        print(metrics["metrics/mAP50-95(B)"])   # Boxen
    - label: Panoptisch
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreEoMTl-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Installation

EoMT benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert. Das Aufgabensuffix im Dateinamen (`-sem`, `-seg`,
`-panoptic`) wählt die Aufgabe aus. `LibreYOLO()` leitet sie aus diesem Namen
ab, sodass kein Argument `task=` erforderlich ist.

<code-tabs name="predict" />

Die semantische Segmentierung füllt `result.semantic_mask`, ein Array der Form
`(H, W)` mit Klassen-IDs in `.data`. Die Instanzsegmentierung füllt
`result.boxes` und `result.masks` in derselben Form wie jede andere
Segmentierungsfamilie. Die panoptische Segmentierung füllt `result.panoptic`:
eine Segment-ID-Karte der Form `(H, W)` in `.data` sowie `.segments_info`, eine
Liste von Dictionaries der Form `{"id", "category_id"}`, eines pro Segment.
`conf` filtert die Query-Auswahl. `iou` hat bei der semantischen Aufgabe keine
Wirkung, weil sie pro Pixel ein Argmax ohne NMS-Schritt berechnet. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt drei Encoder-Größen s/b/l, alle mit DINOv2-Backbone. Der semantische
Checkpoint wurde mit ADE20K bei 512 px trainiert. Die Instanz- und panoptischen
Checkpoints wurden mit COCO bei 640 px trainiert. Ein zweiter Instanz-Checkpoint
wurde bei 1280 px trainiert. Upstream veröffentlicht DINOv2-Gewichte für die
Instanzsegmentierung nur in Größe l. Für die semantische und panoptische
Segmentierung werden s und b veröffentlicht. Upstream bietet auch EoMT-Varianten
mit DINOv3-Backbone an. Sie werden hier nicht ausgeliefert, weil sie von
zugangsbeschränkten, nicht kommerziellen DINOv3-Gewichten abhängen.

LibreYOLO trainiert EoMT nicht: `train()` löst für diese Familie den Fehler
`NotImplementedError` aus. Der oben verlinkte
[Support-Tier](/docs/models) kennzeichnet sie daher als reine Inferenzfamilie.

## Validierung

`val()` verzweigt nach Aufgabe. Die semantische Segmentierung gibt
`metrics/mIoU` und `metrics/pixel_accuracy` zurück. Die Instanzsegmentierung
gibt dieselben mAP-Schlüssel für Masken und Boxen wie andere
Segmentierungsfamilien zurück. Die panoptische Segmentierung gibt Panoptic
Quality als `metrics/PQ` zurück, aufgeteilt in `metrics/SQ`
(Segmentierungsqualität) und `metrics/RQ` (Erkennungsqualität), sowie
`metrics/PQ_things` und `metrics/PQ_stuff`.

<code-tabs name="val" />

## Export

<export-matrix />

Derzeit lässt sich nur die semantische Aufgabe exportieren. Bei Instanz- und
panoptischer Segmentierung löst `export()` den Fehler `NotImplementedError`
aus, weil für ihre Query-Maskenausgabe noch kein Runtime-Exportvertrag besteht.
Ein exportiertes semantisches Artefakt wird anhand seiner Dateiendung wieder
über `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich
daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
