---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: Training, Fine-Tuning und Export unter MIT'
description: >-
  Nutze RF-DETR in LibreYOLO für Erkennung, Instanzsegmentierung, Pose und
  orientierte Boxen. Installiere, sage vorher, trainiere, validiere und
  exportiere, vollständig MIT-lizenziert.
lead: >-
  Ein Detection Transformer, der statt eines dichten Rasters eine feste Menge
  von Objekten vorhersagt und daher bei der Inferenz keine NMS benötigt.
  LibreYOLO unterstützt ihn für vier Aufgaben.
keywords:
  - rf-detr
  - echtzeit detection transformer
  - detr
  - objekterkennung
  - instanzsegmentierung
  - pose schätzung
  - orientierte bounding boxes
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: "LibreRFDETRs, Objekterkennung in einem Video bei 512\_px."
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Video
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")


        # Jede von der Bibliothek unterstützte Quelle: Datei, Ordner, URL,
        Webcam-Index,

        # RTSP-Stream oder eine .streams-Liste

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() gibt ein einfaches dict zurück, kein Objekt
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Gegen COCO
      language: bash
      code: >
        # Die mitgelieferte COCO-YAML enthält ein eingebettetes Downloadskript
        und

        # benötigt daher eine ausdrückliche Erlaubnis, wenn der Datensatz nicht
        lokal ist.

        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)


        # Für jedes Format akzeptierte Argumente:

        #

        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"

        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"

        #             | "tflite" | "coreml" | "coreai".

        #             "engine" ist ein Alias für tensorrt, "litert" für tflite.

        #   imgsz     int oder (height, width). Standardmäßig native
        Checkpoint-Auflösung.

        #   batch     int, Standardwert 1.

        #   half      bool, Export in FP16. Standardwert False.

        #   int8      bool, Export in INT8. Standardwert False. Benötigt `data`.

        #   data      Pfad zu einer Datensatz-YAML zur int8-Kalibrierung.

        #   fraction  float, Anteil des Kalibrierungssatzes. Standardwert 1.0.

        #   dynamic   bool, dynamische Achsen. Standardwert True.

        #   simplify  bool, ONNX-Graphvereinfachung. Standardwert True.

        #   opset     int, ONNX-Opset. Ohne Angabe pro Familie gewählt.

        #   device    str, Gerät für das Tracing. Standardmäßig Modellgerät.

        #   output_path  str, standardmäßig aus dem Checkpoint abgeleitet.

        #   verbose   bool, Standardwert False.

        #   allow_download_scripts  bool, Standardwert False. Erlaubt
        eingebettetes

        #             Python in einer Datensatz-YAML, die heruntergeladen werden
        muss.

        #

        # Einige Formate akzeptieren eigene Zusatzargumente wie eine
        RKNN-Zielplattform.

        # Diese sind auf der jeweiligen Formatseite dokumentiert.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreRFDETRs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
    - label: Ohne LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Bei direkter Graphausführung musst du Vor- und Nachverarbeitung selbst
        umsetzen.

        # Prüfe die Signatur, bevor du etwas anschließt.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Installation

RF-DETR benötigt ein eigenes Zusatzpaket, das `transformers` für das Backbone
installiert.

```bash
pip install "libreyolo[rfdetr]"
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. `conf` und `max_det` filtern die Auswahl der Queries. Es gibt keinen
NMS-Schritt zum Abstimmen. Unter [Vorhersage](/docs/predict) findest du Quellen,
Streaming und die Verarbeitung von Ergebnissen.

## Varianten

Es gibt vier Größen und vier Aufgaben mit einer gemeinsamen Architektur.
Segmentierung, Pose und orientierte Boxen verwenden den Erkennungsdecoder mit
einem anderen Head und akzeptieren daher dieselben Argumente. Die Größen haben
ähnliche Parameteranzahlen und unterscheiden sich vor allem in der
Eingabeauflösung.

<benchmark-table task="detect" />

<va-embed />

## Training

Das Training beginnt bei allen vier Aufgaben mit einem veröffentlichten
Checkpoint. RF-DETR führt `pretrained` unter den von seinem nativen Trainer
ignorierten Argumenten auf. `pretrained=False` erzeugt daher kein zufällig
initialisiertes Modell.

<code-tabs name="train" />

Zwei Argumente sind hier wichtiger als bei einem CNN-Detektor. Halte `lr0` bei
höchstens `1e-4`, weil Transformer-Detektoren bei Lernraten divergieren, die
ein YOLO-Modell verträgt. Behalte für `imgsz` die native Auflösung des
Checkpoints bei, sofern kein Grund für eine Änderung besteht. Die Eingabe muss
ohne Rest durch das Produkt aus Backbone-Patch-Größe und Fensteranzahl teilbar
sein. LibreYOLO prüft das vor Beginn des Laufs und nennt die nächstgelegenen
gültigen Größen.

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

<provenance-box></provenance-box>

## Zitieren

<citation-block />
