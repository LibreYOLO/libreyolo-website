---
title: YOLOv7
families:
  - yolo7
seo_title: 'YOLOv7 in LibreYOLO: Vorhersage, Training und Export unter MIT'
description: >-
  Führe YOLOv7 in LibreYOLO zur Objekterkennung aus: Installation, Vorhersage,
  Training, Validierung und Export mit MIT-lizenziertem Code und Gewichten.
lead: >-
  YOLOv7 ist ein Anchor-basierter Single-Stage-Detektor, dessen Kopf vor der
  letzten Faltung gelernte Offsets aus implizitem Wissen hinzufügt. LibreYOLO
  unterstützt seine einzige veröffentlichte Größe für die Objekterkennung.
keywords:
  - YOLOv7
  - Objekterkennung
  - Anchor-basierte Erkennung
  - implizites Wissen
  - ImplicitA
  - Echtzeit-Objekterkennung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Warmstart mit einem neuen Modell
      language: python
      code: >
        from libreyolo import LibreYOLO7


        # pretrained=True lädt immer den veröffentlichten Checkpoint
        LibreYOLO7b.pt,

        # unabhängig davon, womit diese Instanz erstellt wurde. Bei direkter

        # Erstellung der Klasse statt über LibreYOLO() werden zunächst überhaupt

        # keine Gewichte geladen.

        model = LibreYOLO7(None, size="b")

        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreYOLO7b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Installation

YOLOv7 benötigt neben dem Basispaket kein zusätzliches Extra.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem jeder anderen Familie, sodass der Wechsel zu einem anderen Detektor nur eine einzige Codezeile erfordert. `conf` legt den Konfidenzschwellenwert und `iou` den NMS-Schwellenwert fest, der nach dem Decodieren des Anchor-basierten Kopfes angewendet wird. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

LibreYOLO stellt eine Größe bereit: `b`. Das Upstream-Projekt veröffentlicht ein einziges YOLOv7-Modell, daher gibt es keine Größenauswahl.

## Training

<code-tabs name="train" />

`pretrained` wird tatsächlich ausgewertet, anders als das gleichnamige wirkungslose Argument bei einigen anderen Familien. Übergib `True` für einen Warmstart mit dem veröffentlichten und automatisch heruntergeladenen Checkpoint `LibreYOLO7b.pt` oder einen Pfad beziehungsweise Namen für andere Gewichte. Der veröffentlichte Checkpoint besitzt 80 COCO-Klassen. Wird er für ein bereits auf eine andere Klassenanzahl umgebautes Modell angefordert, baut LibreYOLO zunächst auf 80 Klassen zurück und lädt den Checkpoint. Sobald die Klassenanzahl des Datensatzes bekannt ist, werden anschließend alle Tensoren mit passender Form in den Zielkopf übertragen. `resume=True` lässt sich nicht mit `pretrained` kombinieren. Beim Standardwert `None` wird das Training mit dem Zustand fortgesetzt, mit dem das Modell erstellt wurde, oder ohne geladene Gewichte mit zufälliger Initialisierung.

Ohne weitere Angaben führt der Trainer 300 Epochen mit `lr0=0.01`, SGD-Momentum 0,937, einem Warmup über 3 Epochen sowie derselben SimOTA-Zuweisung und abschließenden 15 Epochen ohne Augmentation wie YOLOX aus. Dies ist an den Anchor-basierten Kopf angepasst. Es gibt einen Unterschied: YOLOX ergänzt in den letzten Epochen eine L1-Verfeinerung der Boxregression, die v7 auslässt, da der SimOTA-Loss von v7 keinen L1-Zweig für Roh-Offsets enthält.

Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Augmentation, Multi-GPU und Loggern.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall, mAP 50 und mAP 50-95 zurück, gemessen anhand jedes Datensatzes im Format, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Du kannst den Graphen auch in einer eigenständigen Laufzeitumgebung ohne installiertes LibreYOLO ausführen. Dann musst du Vor- und Nachverarbeitung selbst implementieren.

<code-tabs name="export" />

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
