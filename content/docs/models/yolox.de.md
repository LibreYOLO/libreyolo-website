---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: Vorhersage, Training und Export unter Apache-2.0'
description: >-
  Verwende YOLOX in LibreYOLO zur Objekterkennung: Installation, Vorhersage,
  Training, Validierung und Export unter Apache-2.0.
lead: >-
  YOLOX ist ein Anchor-freier Single-Stage-Detektor mit entkoppeltem
  Klassifikations- und Regressionskopf, der mit SimOTA-Labelzuweisung trainiert
  wird. LibreYOLO unterstützt ihn für die Objekterkennung.
keywords:
  - YOLOX
  - Objekterkennung
  - Anchor-freie Erkennung
  - entkoppelter Kopf
  - SimOTA
  - Echtzeit-Objekterkennung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Gegen COCO
      language: bash
      code: >
        # Die mitgelieferte COCO-YAML-Datei enthält ein Downloadskript und
        benötigt

        # daher eine ausdrückliche Berechtigung, sofern der Datensatz noch nicht
        lokal vorliegt.

        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreYOLOXs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Installation

YOLOX benötigt neben dem Basispaket kein zusätzliches Extra.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem jeder anderen Familie, sodass der Wechsel zu einem anderen Detektor nur eine einzige Codezeile erfordert. `conf` legt den Konfidenzschwellenwert und `iou` den NMS-Schwellenwert fest, die auf die drei entkoppelten Vorhersageskalen angewendet werden. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Sechs Größen verwenden denselben CSP-Backbone und PAFPN-Neck. Die beiden kleinsten, `n` und `t`, werden mit einer kleineren festen Eingabeauflösung als die übrigen vier ausgeführt. Die folgende Benchmark-Tabelle enthält den genauen Wert für jede Größe.

<benchmark-table task="detect" />

<va-embed />

## Training

<code-tabs name="train" />

Ohne weitere Angaben führt der Trainer 300 Epochen mit `lr0=0.01`, SGD-Momentum 0,9 und einem Warmup über 5 Epochen aus. Mosaic- und MixUp-Augmentation werden für die letzten 15 Epochen deaktiviert. `train()` akzeptiert außerdem ein Argument `pretrained`, dessen Wert innerhalb der Methode jedoch nie gelesen wird. Das Training wird immer mit den Gewichten fortgesetzt, mit denen das Modell erstellt wurde. `pretrained=False` initialisiert das Netzwerk daher nicht neu.

`imgsz` verwendet standardmäßig einen festen Wert aus der Basiskonfiguration des Trainings und nicht die native Auflösung des geladenen Checkpoints. Das betrifft insbesondere die Checkpoints `n` und `t`: Wenn du ihr Training ohne explizite Angabe von `imgsz` fortsetzt, wechseln sie von der kleineren veröffentlichten Größe zum größeren Standardwert.

Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Augmentation, Multi-GPU und Loggern.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall, mAP 50 und mAP 50-95 zurück, gemessen anhand jedes Datensatzes im Format, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Du kannst den Graphen auch in einer eigenständigen Laufzeitumgebung ohne installiertes LibreYOLO ausführen. Dann musst du Vor- und Nachverarbeitung selbst implementieren. Ein CoreML-Export kann mit `nms=True` NMS in den Graphen einbetten. YOLOX und YOLOv9 sind derzeit die einzigen beiden Familien, die dieses Flag akzeptieren.

<code-tabs name="export" />

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
