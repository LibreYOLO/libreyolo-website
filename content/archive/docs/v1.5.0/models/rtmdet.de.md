---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet in LibreYOLO: Vorhersage, Training und Export'
description: >-
  Führe RTMDet in LibreYOLO für Objekterkennung und
  RTMDet-Ins-Instanzsegmentierung aus. Installiere, sage vorher, trainiere,
  validiere und exportiere unter Apache-2.0.
lead: >-
  RTMDet ist ein einstufiger Detektor, der von einem punktbasierten Prior pro
  Rasterposition ohne Anchors vorhersagt. Die Faltungen seines Heads werden über
  die Feature-Ebenen hinweg gemeinsam verwendet. LibreYOLO unterstützt
  Objekterkennung und RTMDet-Ins-Instanzsegmentierung.
keywords:
  - rtmdet python
  - objekterkennung
  - instanzsegmentierung
  - rtmdet-ins
  - ankerfreie objekterkennung
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Instanzsegmentierung
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -seg im Dateinamen wählt den RTMDet-Ins-Masken-Head aus,
        # daher ist hier kein task-Argument erforderlich.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Instanzsegmentierung
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # Masken
        print(metrics["metrics/mAP50-95(B)"])   # Boxen
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreRTMDets.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Installation

RTMDet benötigt neben dem Basispaket kein Zusatzpaket.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. Ein Dateiname mit `-seg` wird selbstständig der RTMDet-Ins-Aufgabe
zugeordnet. `result.masks` enthält dann die Instanzmasken neben den Boxen.
`conf` legt den Confidence-Schwellenwert und `iou` den NMS-Schwellenwert fest.
Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

Es gibt fünf Größen von `t` bis `x`, die eine Architektur und eine gemeinsame
Eingabeauflösung verwenden. Für diese Familie gibt es hier keine
Benchmark-Tabelle. Vergleiche die Größen anhand der Checkpoint-Dateigröße in
der folgenden Tabelle.

## Training

<code-tabs name="train" />

Die Objekterkennung wird über `train()` trainiert. Die Komponenten
QualityFocalLoss, GIoU und DynamicSoftLabelAssigner wurden aus dem Upstream
mmdetection portiert. Vorwärtslauf und ONNX-Export sind bitgleich. Die
Nachverarbeitung stimmt auf Teilmengen von val2017 innerhalb von 0.001 mAP mit
der mmdet-Ausgabe überein.

Laut dem eigenen Docstring von `train()` wurde Folgendes nicht geprüft:
Fine-Tuning-Konvergenz auf kleinen Datensätzen, Parität mit der Veröffentlichung
beim Training von Grund auf neu, Multi-GPU-Verhalten, Durchsatz des
zwischengespeicherten Mosaic und MixUp, der strikte zweistufige
Upstream-Pipeline-Wechsel und die parameterweisen Weight-Decay-Ausnahmen, die
Decay für Normalisierungs- und Bias-Parameter auf null setzen.

RTMDet-Ins besitzt keinen Trainingspfad. Ein Aufruf von `train()` auf einem
`-seg`-Checkpoint oder mit `task="segment"` löst `NotImplementedError` aus.
Die Instanzsegmentierung unterstützt nur Inferenz und Validierung.

`train()` akzeptiert außerdem ein Argument `pretrained`, liest dessen Wert aber
nie in der Methode. Das Training wird immer mit den Gewichten fortgesetzt, mit
denen das Modell erzeugt wurde. `pretrained=False` initialisiert das Netzwerk
daher nicht neu.

Ansonsten läuft der Trainer ohne Änderungen 300 Epochen mit AdamW bei
`lr0=0.004` und `weight_decay=0.05`, einem Warmup von 1 Epoche auf einem
Cosine-Zeitplan sowie in den letzten 20 Epochen deaktiviertem Mosaic und MixUp.

Unter [Training](/docs/train) findest du Datensätze, Datenaugmentierung,
Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

<code-tabs name="val" />

Bei einem `-seg`-Checkpoint enthält der einfache Schlüssel
`metrics/mAP50-95` den Masken-Score. Derselbe Lauf meldet außerdem Boxen mit
dem Suffix `(B)` und Masken mit `(M)`, sodass beide in einem Durchlauf verfügbar
sind.

## Export

<export-matrix />

Die Objekterkennung lässt sich in die meisten Formate exportieren. Die
Instanzsegmentierung derzeit in keines. Die obige Matrix bildet diese Trennung
ab. Ein exportiertes Erkennungsartefakt wird anhand seiner Dateiendung wieder
über `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich wie
ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Du kannst den Graphen
auch in einer einfachen Runtime ohne LibreYOLO ausführen. Dann musst du die
Vor- und Nachverarbeitung selbst implementieren.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
