---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: Vorhersage, Training und Export unter MIT'
description: >-
  Führe YOLOv9 in LibreYOLO aus, einschließlich NMS-freiem End-to-End-Kopf und
  Stride-4-Kopf für kleine Objekte. Installation, Vorhersage, Training,
  Validierung und Export.
lead: >-
  Ein konvolutionaler Single-Stage-Detektor: Ein Durchlauf bewertet ein dichtes
  Raster von Boxen und NMS entfernt Duplikate. LibreYOLO bietet drei Varianten,
  von denen eine ohne NMS-Schritt auskommt.
keywords:
  - YOLOv9
  - YOLO9
  - Objekterkennung
  - NMS-freie Erkennung
  - End-to-End-Objekterkennung
  - Erkennung kleiner Objekte
  - Programmable Gradient Information
  - GELAN
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Ohne NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Derselbe Aufruf mit einem anderen Checkpoint. Der End-to-End-Kopf
        liefert

        # seine am höchsten bewerteten Vorhersagen selbst, daher wird kein NMS
        ausgeführt und iou ignoriert.

        model = LibreYOLO("LibreYOLO9E2Es.pt")

        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)


        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Kleine Objekte
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # Die Stride-4-Variante besitzt keinen eigenen COCO-Checkpoint. Gib
        daher einen

        # Basischeckpoint zur Objekterkennung an: Backbone und Neck werden
        unverändert geladen,

        # während der Stride-4-Kopfturm zufällig initialisiert wird.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Gegen COCO
      language: bash
      code: >
        # Die mitgelieferte COCO-YAML-Datei enthält ein Downloadskript und
        benötigt

        # daher eine ausdrückliche Berechtigung, sofern der Datensatz noch nicht
        lokal vorliegt.

        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: Mit NMS im Graphen
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreYOLO9s.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c322dd68d42cdf24
---

## Installation

YOLOv9 benötigt neben dem Basispaket kein zusätzliches Extra.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem jeder anderen Familie, sodass der Wechsel zu einem anderen Detektor nur eine einzige Codezeile erfordert. Bei den Basismodellen und den Stride-4-Modellen legt `conf` den Konfidenzschwellenwert und `iou` den NMS-Schwellenwert fest. Das End-to-End-Modell führt kein NMS aus und ignoriert `iou`, daher bestimmen `conf` und `max_det` seine Ausgabe. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Drei Varianten verwenden denselben Backbone. Alle drei sind ausschließlich für die Objekterkennung vorgesehen und akzeptieren dieselben Argumente.

Das Basismodell sagt auf drei Merkmalsskalen vorher und entfernt doppelte Boxen mit NMS.

Das End-to-End-Modell behält diesen Kopf bei und ergänzt daneben einen Zweig mit One-to-One-Zuordnung. Die Inferenz liest ausschließlich diesen Zweig und übernimmt dessen am höchsten bewertete Vorhersagen, weshalb kein NMS ausgeführt wird. Wähle diese Variante, wenn deine Ziellaufzeitumgebung keinen NMS-Operator besitzt.

Das Stride-4-Modell greift eine Ebene weiter oben im Backbone ab, erweitert den Neck bis zu dieser Ebene und sagt auf vier statt drei Skalen vorher. Die zusätzliche Skala ist für Objekte gedacht, die nur wenige Pixel belegen. Der einzige dafür veröffentlichte Checkpoint wurde auf Luftbildern trainiert. Basischeckpoints zur Objekterkennung lassen sich darauf übertragen. Backbone und Neck werden unverändert geladen, die drei vortrainierten Kopftürme rücken um einen Platz nach oben und der Stride-4-Turm wird zufällig initialisiert.

<benchmark-table task="detect" />

<va-embed />

## Training

<code-tabs name="train" />

`pretrained` bestimmt den Ausgangspunkt des Laufs. Übergib `True`, um den veröffentlichten Checkpoint für dasselbe Modell und dieselbe Größe zu laden, oder einen Namen beziehungsweise Pfad für andere Gewichte. Tensoren mit unpassender Form werden übersprungen statt abgelehnt. Der Lauf protokolliert die Anzahl der geladenen Tensoren, sodass auch ein mit einer anderen Klassenanzahl trainierter Checkpoint als Ausgangspunkt dienen kann.

Das Stride-4-Modell besitzt keinen eigenen veröffentlichten COCO-Checkpoint. `True` wird dafür zu einer nicht vorhandenen Datei aufgelöst und der Download schlägt fehl. Gib stattdessen einen Basischeckpoint zur Objekterkennung an.

Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Augmentation, Multi-GPU und Loggern.

Das Basismodell kann neben seinem Haupt-Head einen PGI-Hilfszweig trainieren. Der Zweig wird nur beim Training verwendet: Vorhersage und Export nutzen den Haupt-Head. Ein Lauf hängt ihn an, wenn die Startgewichte PGI-Tensoren enthalten oder das Training von Grund auf neu beginnt. Die veröffentlichten Basis-Checkpoints enthalten keine, daher trainiert ein Fine-Tuning von ihnen nur den Haupt-Head und protokolliert das. `aux_weight=0.25` hängt trotzdem einen neuen Zweig mit diesem Loss-Gewicht an, `aux_weight=0` hält ihn ausgeschaltet. Ein fortgesetzter Lauf behält das PGI-Setup seines Checkpoints. Bei den Größen `m` und `c` ist der Zweig der aus dem Upstream: ein zweites Backbone auf dem Eingabebild, dessen Stufen `CBFuse` mit `CBLinear`-Abgriffen des Haupt-Backbones aufsummiert, und ein eigener Head, bei 80 Klassen 12.8M und 25.7M Parameter, die nur im Training existieren. Das End-to-End-Modell und das Stride-4-Modell haben keinen PGI-Zweig.

Alle drei Varianten beschneiden die L2-Norm des Gradienten vor jedem Optimizer-Schritt auf 10; `clip_max_norm=0` schaltet das Clipping ab. Die Label-Zuweisung bewertet Anchors in fp32, auch unter float16-Autocast. `max_labels=300` begrenzt die pro Trainingsbild behaltenen Boxen, darüber hinausgehende Boxen werden mit einer einmaligen Warnung verworfen; der Speicherbedarf der Zuweisung folgt den tatsächlichen Boxen jedes Bildes, nicht der Obergrenze. Das SGD-Momentum steigt während drei Epochen von 0.8 auf 0.937. `letterbox_pad=None` übernimmt den Checkpoint-Eintrag: Nicht markierte Gewichte verwenden `topleft`, neue offizielle Konvertierungen speichern `center`.

Mosaic in YOLO9 und YOLOX bevorzugt annotierte Partnerbilder bei höchstens 20 Ziehungen; YOLO9-MixUp verwendet dieselbe Regel. MixUp ist standardmäßig aus (`mixup_prob=0`). Ist es an, wird das Mischverhältnis aus Beta(1, 1) gezogen, gleichverteilt auf [0, 1], wie in MultimediaTechLab/YOLO. Siehe [Ereignishistogramme](/docs/train/event-histograms) für Eingabeprofile ohne RGB.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall, mAP 50 und mAP 50-95 zurück, gemessen anhand jedes Datensatzes im Format, das du für das Training verwendet hast.

<code-tabs name="val" />

## Export

<export-matrix />

Ein Häkchen gilt für alle drei Varianten. Wo sie sich unterscheiden, zeigt die Matrix die schwächste Unterstützung der drei.

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Du kannst den Graphen auch in einer eigenständigen Laufzeitumgebung ohne installiertes LibreYOLO ausführen. Dann musst du Vor- und Nachverarbeitung selbst implementieren.

Beim Basismodell zur Objekterkennung lässt sich die Nachverarbeitung in den Graphen verschieben. `nms=True` bei einem ONNX-Export bettet die Unterdrückung in das Modell ein. Die erste Ausgabe wird zu einem festen Tensor der Form `(1, max_det, 6)`, dessen Zeilen `x1, y1, x2, y2, score, class` enthalten und hinter der Anzahl der Erkennungen mit Nullen aufgefüllt werden. Dieser Graph besitzt Batchgröße 1 und keine dynamischen Achsen. Das End-to-End-Modell und das Stride-4-Modell akzeptieren dieses Flag nicht.

Jedes Format installiert ein anderes Extra und besitzt eigene Argumente. Beides ist auf der Seite des jeweiligen Formats beschrieben.

<code-tabs name="export" />

[TFLite INT8](/docs/export/tflite) verwendet `int8=True` mit Kalibrierungsdaten.

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Der Code aller drei Varianten ist aus
[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) beim Commit
`c4cb5f6f` (MIT) portiert. Der One-to-One-Prädiktor des End-to-End-Modells folgt
dem Design von [DATE](https://github.com/YiqunChen1999/date) von Yiqun Chen
(Apache-2.0), für LibreYOLO implementiert, ohne dass DATE-Code kopiert wurde. Die
Stride-4-Ebene ist LibreYOLO-Code.

Der Luftbild-Checkpoint mit Stride 4 sagt die VisDrone-Klassen vorher. Verwende die vom Herausgeber im Gewichtsrepository deklarierte Lizenz.

</provenance-box>

## Zitieren

<citation-block />
