---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet in LibreYOLO: Vorhersage, Training und Export'
description: >-
  Führe PicoDet in LibreYOLO für mobile Objekterkennung aus. Installiere, sage
  vorher, trainiere, validiere und exportiere unter Apache-2.0.
lead: >-
  PicoDet ist ein einstufiger Detektor für Mobil- und Edge-CPUs: ein
  ESNet-Backbone, ein CSP-PAN-Neck und ein gemeinsamer
  Generalized-Focal-Loss-Head. LibreYOLO unterstützt ihn für die
  Objekterkennung.
keywords:
  - picodet python
  - pp-picodet
  - objekterkennung
  - mobile objekterkennung
  - edge detection
  - esnet
  - generalized focal loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz sollte gesetzt werden: Die CLI nutzt standardmäßig 640,

        # der s-Checkpoint ist nativ für 320 ausgelegt.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibrePICODETs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Installation

PicoDet benötigt neben dem Basispaket kein Zusatzpaket.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. `conf` legt den Confidence-Schwellenwert und `iou` den NMS-Schwellenwert
fest. Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

Es gibt drei Größen mit jeweils eigener fester Eingabeauflösung: `s` ist die
kleinste und `l` die größte. Die Auflösung wächst mit der Größe. Größere
Checkpoints benötigen pro Bild zusätzlich zu ihrer höheren Parameteranzahl
mehr Rechenleistung.

<benchmark-table task="detect" />

<va-embed />

## Training

<code-tabs name="train" />

Loss-Komponenten und Assigner entsprechen dem Upstream-Rezept: VFL, DFL, GIoU
und SimOTA mit Gewichtung nach Klassifizierungsqualität und dynamischen
IoU-VFL-Zielen. Die Inferenz ist auf demselben Checkpoint bitgleich mit dem
Upstream.

Laut dem eigenen Docstring von `train()` wurde Folgendes nicht geprüft:
Konvergenz auf einem vollständigen Datensatz, Multi-GPU-Verhalten und jede
Augmentierung außer horizontaler Spiegelung. Der Checkpoint `s` bei seiner
nativen Größe 320 hat außerdem die Accuracy-Untergrenze von LibreYOLO auf dem
Testdatensatz mit 30 Bildern und zwei Klassen für kleine Fine-Tunings nicht
zuverlässig überschritten. Diese Größe eignet sich besser im vollen
COCO-Maßstab.

`train()` akzeptiert auch das Argument `pretrained`, liest dessen Wert aber
nie innerhalb der Methode. Das Training wird immer mit den Gewichten
fortgesetzt, mit denen das Modell erzeugt wurde. `pretrained=False`
initialisiert das Netzwerk daher nicht neu. Wenn du `imgsz` in Python nicht
festlegst, wird die native Auflösung des geladenen Checkpoints verwendet: 320
für `s`, 416 für `m` und 640 für `l`. Die CLI sendet immer einen Wert für
`imgsz`, standardmäßig 640. Lege ihn dort passend zum Checkpoint fest.

Ansonsten läuft der Trainer ohne Änderungen 300 Epochen mit SGD bei `lr0=0.01`,
Momentum 0.9, Weight Decay 4e-5 und einem Warmup von 1 Epoche auf einem
Cosine-Zeitplan. Die horizontale Spiegelung ist die einzige Augmentierung.

Unter [Training](/docs/train) findest du Datensätze, Datenaugmentierung,
Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

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

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Die LibreYOLO-Portierung folgt Bo396543018/Picodet_Pytorch, einer
PyTorch-Neuimplementierung des ursprünglichen PP-PicoDet aus PaddleDetection.
mmcv wurde entfernt, und jede Aktivierung wurde exakt abgeglichen. Dadurch
werden über Bos Pipeline konvertierte PaddlePaddle-Checkpoints ohne numerische
Abweichung geladen. Beide Quellen stehen unter denselben Apache-2.0-Bedingungen
wie die Autoren der Veröffentlichung.

</provenance-box>

## Zitieren

<citation-block />
