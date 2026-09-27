---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: Texterkennung und -erfassung in LibreYOLO'
description: >-
  Nutze PP-OCRv5 in LibreYOLO für mehrsprachige OCR von Szenentext. Installiere,
  sage vorher und validiere die Checkpoints t und l unter Apache-2.0.
lead: >-
  PP-OCRv5 ist die Pipeline von PaddleOCR für Texterfassung und -erkennung: Ein
  Detektor mit differenzierbarer Binarisierung lokalisiert Textvierecke, ein
  SVTR/CTC-Erkenner liest sie. LibreYOLO portiert zwei Tiers nach PyTorch.
keywords:
  - pp-ocrv5
  - paddleocr
  - ocr python
  - texterfassung
  - texterkennung
  - szenentext erkennen
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Vierecke
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibrePPOCRl-ocr.pt")

        result = model(SAMPLE_IMAGE)


        # Polygone der Form (N, 4, 2) in Lesereihenfolge: oben links, oben
        rechts,

        # unten rechts, unten links. Erkennungsvierecke sind echte Polygone

        # (gedrehter Text) und landen daher in result.ocr statt result.boxes.

        print(result.ocr.data.shape)

        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # Hauptmetrik
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Installation

PP-OCRv5 benötigt neben dem Basispaket kein Zusatzpaket.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Jeder Checkpoint bündelt beide Phasen, Erfassung und Erkennung, in einer
`.pt`-Datei. Zeichensatz und Pipeline-Standardwerte des Erkenners stehen in den
Checkpoint-Metadaten. Der Erkenner liest vereinfachtes und traditionelles
Chinesisch, Englisch, Japanisch und Pinyin mit einem Wörterbuch. `result.ocr`
ist eine `OCRRegions`-Nutzlast: `.data` enthält die Vierpunktpolygone, `.texts`
die Transkripte, `.conf` den Erkennungs-Score pro Region und `.det_conf` den
Erfassungs-Score. Quellen mit mehreren Bildern werden nacheinander verarbeitet.
Die zweistufige Pipeline bildet keine Batches über mehrere Bilder. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt zwei Tiers: `t` verwendet leichtere PP-LCNetV3/PP-OCRv5_mobile-Backbones
für CPUs, `l` PP-HGNetV2-Server-Backbones für höhere Accuracy. Beide Tiers
führen die Erfassung mit einer festen Grenze für die lange Seite aus und
erkennen Zuschnitte in Batches. `rec_batch` steuert, wie viele Zuschnitte pro
Vorwärtslauf den Erkenner durchlaufen.

## Validierung

`val()` misst die Pipeline anhand eines Bildverzeichnisses plus einer Datei
`labels/<split>.jsonl` oder einer gleichwertigen Datensatz-YAML. Jedes Label
führt die Polygone und Transkripte der Textregionen eines Bildes auf. Gemeldet
werden Detection Hmean (Precision/Recall/F1 mit IoU-Abgleich), End-to-End F1
(Hmean plus exakte Übereinstimmung des normalisierten Transkripts, die
Fitnessmetrik des Checkpoints) und 1-NED, die mittlere normalisierte
Editierdistanz über zugeordnete Paare.

<code-tabs name="val" />

## Export

<export-matrix />

PP-OCRv5 ist eine Pipeline aus zwei Netzwerken, Erfassung und Erkennung, die
gemeinsam verschoben werden, und kein einzelner tracebarer Graph. Der Export
ist nicht implementiert, noch wird ein Format unterstützt. Trainiere den
Apache-2.0-Upstream-Code direkt nach und konvertiere das Ergebnis mit
`weights/convert_ppocr_weights.py`, wenn du einen Checkpoint außerhalb dieses
Formats benötigst.

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
