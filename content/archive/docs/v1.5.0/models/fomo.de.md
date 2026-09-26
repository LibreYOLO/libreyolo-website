---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: Punktlokalisierung, Training und Export in LibreYOLO'
description: >-
  Führe FOMO (Faster Objects, More Objects) in LibreYOLO aus: einen winzigen
  Punktlokalisierungsdetektor zum Zählen vieler kleiner Objekte. Installiere,
  sage vorher, trainiere und exportiere.
lead: >-
  FOMO ist ein rasterbasierter Punktlokalisierer: Jede Zelle eines niedrig
  aufgelösten Rasters wird als Hintergrund oder Objektzentrum klassifiziert,
  ohne Bounding-Box-Regression. LibreYOLO unterstützt das Modell für die
  Punktaufgabe.
keywords:
  - fomo objekterkennung
  - faster objects more objects
  - punktlokalisierung
  - objektzentren erkennen
  - winzige objekte erkennen
  - edge ai
  - mcu objekterkennung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # LibreFOMO-Gewichte werden nicht automatisch geladen (siehe Checkpoints
        unten).

        # Verweise hier auf einen bereits lokal heruntergeladenen Checkpoint.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz muss übergeben werden: Die CLI nutzt standardmäßig 640, der

        # s-Checkpoint akzeptiert nur seine native Größe 96.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Installation

FOMO benötigt neben dem Basispaket kein Zusatzpaket.

```bash
pip install libreyolo
```

## Vorhersage

Anders als bei jeder anderen Familie auf dieser Website werden LibreFOMO-Gewichte
nicht automatisch heruntergeladen: `LibreYOLO("LibreFOMOs-point.pt")` sucht
auf dem Datenträger nach dieser Datei und löst einen `ValueError` aus, der sie
benennt, statt sie von Hugging Face abzurufen. Lade zunächst einen Checkpoint
von der [LibreYOLO-Organisation](https://huggingface.co/LibreYOLO) herunter und
lade ihn über seinen lokalen Pfad. Alternativ kannst du einen eigenen trainieren
(siehe Training unten).

<code-tabs name="predict" />

Das Ergebnis enthält eine `points`-Nutzlast statt `boxes`: Jede Zeile besteht
aus `x, y, class, confidence` und ist als `result.points.data` oder über die
Zugriffsfunktionen `.xy`, `.xyn`, `.cls` und `.conf` verfügbar. Es gibt keinen
`iou`-Schwellenwert, weil keine Boxen unterdrückt werden. Mit
`predict(..., nms_radius=1)` steuerst du, wie viele Rasterzellen zwei
Erkennungen auseinanderliegen müssen, damit beide erhalten bleiben. Der
Dateiname muss das FOMO-Aufgabensuffix `-point` enthalten, damit der Loader ihn
erkennt. Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und
die Verarbeitung von Ergebnissen.

## Varianten

Die drei Größen `s`, `m` und `l` verwenden zunehmend breitere Backbones im
MobileNetV2-Stil mit entsprechend größeren festen Eingabeauflösungen. Dahinter
liegt jeweils ein einzelner 1x1-Klassifizierungs-Head. Für diese Familie gibt
es hier keine Benchmark-Tabelle. Die Dateigröße der Checkpoints in der
folgenden Tabelle ist derzeit das deutlichste veröffentlichte Signal pro Größe.

## Training

<code-tabs name="train" />

`imgsz` ist nicht frei wählbar. Standardmäßig wird die native Auflösung des
geladenen Checkpoints verwendet. Die Übergabe eines anderen Werts löst einen
`ValueError` aus, der die erwartete Größe nennt. Diese Größen sind 96 für `s`,
192 für `m` und 224 für `l`. Die CLI verwendet standardmäßig 640 für `imgsz`.
Ein Befehl `libreyolo train` muss den Wert deshalb ausdrücklich passend zum
Checkpoint festlegen.

Ansonsten läuft der Trainer ohne Änderungen 40 Epochen mit Batch 32, Adam bei
`lr0=3e-4`, ohne Weight Decay und mit einer gegenüber dem Hintergrund um den
Faktor 100 gewichteten Vordergrundklasse im zellweisen Cross-Entropy-Loss. In
einer typischen Szene gehört nahezu jede Rasterzelle zum Hintergrund. EMA und
Mixed Precision sind standardmäßig deaktiviert. Keine der sonst in LibreYOLO
verwendeten geometrischen oder farblichen Augmentierungen wird angewendet:
Mosaic, Mixup, HSV-Jitter, Spiegelung, Rotation, Translation und Scherung sind
alle null.

Über diesen Pfad wurden die veröffentlichten LibreFOMO-Checkpoints von Grund
auf neu auf COCO trainiert.

Unter [Training](/docs/train) findest du Datensätze und Logger.

## Validierung

`val()` verwendet einen für diese Familie entwickelten Validator auf
Rasterebene. Neben den Schlüsseln `metrics/precision`, `metrics/recall` und
`metrics/mAP@` aus dem Punktabgleich, die auch andere Punktaufgaben verwenden,
testet er Confidence-Schwellenwerte und Werte für `nms_radius`. Die Kombination
mit dem besten F1-Wert wird unter `metrics/grid_F1`, `metrics/grid_precision`,
`metrics/grid_recall` und `metrics/grid_mean_distance` veröffentlicht. Der
zugehörige Schwellenwert und Radius stehen unter `decode/threshold` und
`decode/nms_radius`.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück. Du kannst den Graphen auch in einer
einfachen Runtime ohne LibreYOLO ausführen. Dann musst du die Vor- und
Nachverarbeitung selbst implementieren.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie. Keine davon wird
automatisch heruntergeladen. Rufe die gewünschte Datei von der verlinkten
Hugging-Face-Seite ab und übergib ihren lokalen Pfad an `LibreYOLO()`.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Es gibt kein Upstream-Code-Repository für FOMO, das verlinkt werden könnte.
Edge Impulse beschreibt die Methode in einem Blogbeitrag und seiner
Produktdokumentation, hat aber keinen Trainings- oder Inferenzcode für FOMO
veröffentlicht. Die Architektur und das Training hier sind die eigene
Implementierung dieser veröffentlichten Beschreibung durch LibreYOLO. Die
veröffentlichten LibreFOMO-Checkpoints wurden von Grund auf neu auf COCO
trainiert. Code und Gewichte stehen daher unter der LibreYOLO-eigenen
MIT-Lizenz. Der Name FOMO und die dadurch bezeichnete Methode gehören weiterhin
zu Edge Impulse.

</provenance-box>
