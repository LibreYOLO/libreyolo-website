---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: erkennen, Posen schätzen und segmentieren in LibreYOLO'
description: >-
  Nutze EdgeCrafter in LibreYOLO für Objekterkennung, Pose-Schätzung und
  Instanzsegmentierung. Installiere, sage vorher, validiere und exportiere mit
  MIT-lizenziertem Code.
lead: >-
  Ein kompakter Vision Transformer für dichte Vorhersagen auf Edge-Hardware, der
  upstream als drei verwandte Modelle veröffentlicht wurde: ECDet, ECPose und
  ECSeg. LibreYOLO lädt alle drei als eine Familie, wobei der Checkpoint die
  Aufgabe festlegt.
keywords:
  - edgecrafter
  - ecdet
  - ecpose
  - ecseg
  - kompakter vision transformer
  - objekterkennung
  - pose schätzung
  - instanzsegmentierung
  - edge inference
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -pose im Dateinamen wählt den Keypoint-Head aus,
        # daher ist hier kein task-Argument erforderlich.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Instanzsegmentierung
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Benötigt einen einklassigen Keypoint-Datensatz, dessen data.yaml
        # kpt_shape deklariert, und imgsz in der nativen Größe des Checkpoints.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Instanzsegmentierung
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Benötigt Polygon-Labels und imgsz in der nativen Größe des
        Checkpoints.

        model = LibreYOLO("LibreECs-seg.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Instanzsegmentierung
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # Masken
        print(metrics["metrics/mAP50-95(B)"])   # Boxen
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreECs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## Installation

EdgeCrafter benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

Eine Ausnahme ist das Adapter-Fine-Tuning mit `lora=True`. Dafür ist das
Zusatzpaket `lora` erforderlich.

```bash
pip install "libreyolo[lora]"
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Die Aufgabe stammt aus dem Dateinamen. Ein `-pose`- oder `-seg`-Checkpoint
wählt somit seinen eigenen Head aus und benötigt kein Aufgabenargument. Alle
drei geben das `Results`-Objekt aller Familien zurück, ergänzt um
`result.keypoints` für Pose und `result.masks` für Segmentierung. Die
Pose-Schätzung umfasst eine Klasse, Person, mit den 17 COCO-Keypoints. Ihre
Anzahl wird beim Erstellen des Modells festgelegt. Das Modell besitzt keinen
Box-Head. Jede Pose-Box entspricht daher der Bounding Box ihrer eigenen
Keypoints, und der dritte Keypoint-Kanal ist eine Konstante statt eines
punktbezogenen Scores.

`conf` und `max_det` filtern die Auswahl der Queries. `iou` wird aus Gründen
der API-Parität akzeptiert, hat aber keine Wirkung, weil alle drei Heads eine
Menge von Queries ohne NMS-Schritt dekodieren. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt vier Größen. Alle laufen mit derselben Eingabeauflösung, daher
unterscheidet die Tabelle sie anhand der Parameteranzahl und Accuracy.

<benchmark-table task="detect" />

<va-embed />

Upstream veröffentlicht ECDet, ECPose und ECSeg als drei separate Modelle und
nicht als ein Modell mit drei Heads. Sie verwenden dasselbe ECViT-Backbone und
denselben hybriden Encoder und unterscheiden sich nur im Head. LibreYOLO fasst
sie deshalb in einer Familie zusammen und lässt den Checkpoint-Dateinamen die
Aufgabe tragen. Ein Größenbuchstabe bezeichnet somit bei allen drei Modellen
dasselbe Backbone und denselben Encoder. Vorhersage, Validierung und Export
akzeptieren unabhängig vom geladenen Modell dieselben Argumente.

## Training

Alle drei Aufgaben werden über `train()` trainiert. Die Methode liest die
Aufgabe aus dem geladenen Checkpoint und wählt den passenden Trainer.

<code-tabs name="train" />

Für Erkennung und Segmentierung wurde Folgendes geprüft: Inferenzparität mit
dem Upstream auf 1e-5, Schicht für Schicht und für jede Größe, sowie die
Ausführung des Losses und eines einzelnen Trainingsschritts auf synthetischer
Eingabe. Laut dem eigenen Docstring von `train()` wurde Folgendes nicht geprüft:
die Konvergenz eines vollständigen Fine-Tunings, Multi-GPU-Training, der Schritt
zum Neuladen des besten Modells nach dem Stopp der Augmentierung und die
Klassenumordnung von Objects365 zu COCO. Der Pose-Pfad folgt dem veröffentlichten
Rezept von DETRPose: einem ungarischen Matcher über Klassen-, Keypoint-L1- und
OKS-Kosten mit kontrastivem Keypoint-Denoising. Auch seine Konvergenz wurde
nicht vollständig geprüft.

Ohne Änderungen läuft der Trainer 74 Epochen mit `lr0=5e-4` und aktivierter
Mixed Precision gemäß dem Upstream-Rezept: AdamW, ein Flat-Cosine-Zeitplan,
EMA mit 0.9999 und nach ImageNet normalisierte Eingaben. Pose und Segmentierung
benötigen beide `imgsz` in der nativen Größe des Checkpoints, weil ihr
Auswertungs-Anchor-Raster beim Erstellen des Modells aufgebaut wird. Ein anderer
Wert löst vor Beginn des Laufs einen Fehler aus. Pose benötigt außerdem einen
einklassigen Datensatz, dessen `data.yaml` den Wert `kpt_shape` deklariert. Die
Keypoint-Anzahl muss zum Head passen.

`lora=True` gilt nur für die Objekterkennung. Pose und Segmentierung lösen dafür
einen `ValueError` aus. Auf Apple-Silicon verbleibt der Lauf auf der GPU, der
Trainer verlagert aber eine Operation auf die CPU: den Grid-Sample-Backward-Pass
innerhalb der Deformable Attention, den PyTorch nicht in Metal implementiert.

Unter [Training](/docs/train) findest du Datensätze, Datenaugmentierung,
Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit den Metriknamen als Schlüsseln zurück und
druckt klassenspezifische Ergebnisse, solange `verbose` aktiviert bleibt.

<code-tabs name="val" />

Pose meldet Keypoint-OKS-Metriken unter `metrics/keypoints_*`. Die Segmentierung
meldet Masken unter dem einfachen Schlüssel `metrics/mAP50-95` und wiederholt
beide Ansichten in einem Durchlauf: Boxen unter `(B)` und Masken unter `(M)`.

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück. Pose und Segmentierung werden mit
einer festen Eingabe von 640 mal 640 statt mit dynamischen Formen exportiert.
Auch mehrere Erkennungsziele verwenden eine feste Arbeitsfläche, darunter
OpenVINO, Paddle, MNN, ExecuTorch und Core AI. [Export](/docs/export) führt die
Argumente auf, die jedes Format akzeptiert, sowie die Zusatzpakete, die einige
davon benötigen.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
