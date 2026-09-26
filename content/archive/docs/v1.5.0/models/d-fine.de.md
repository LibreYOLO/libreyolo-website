---
title: D-FINE
families:
  - dfine
seo_title: 'D-FINE: Fine-Tuning, Validierung und Export unter MIT'
description: >-
  Nutze D-FINE in LibreYOLO für Objekterkennung und Instanzsegmentierung.
  Installation, Vorhersage, Fine-Tuning, Validierung und Export, mit
  MIT-lizenziertem Code.
lead: >-
  Ein Detection Transformer, der die Box-Regression als
  Wahrscheinlichkeitsverteilung über jede Box-Kante formuliert und über die
  Decoder-Schichten hinweg verfeinert. LibreYOLO unterstützt ihn für
  Objekterkennung und Instanzsegmentierung.
keywords:
  - D-FINE
  - detection transformer
  - echtzeit objekterkennung
  - instanzsegmentierung python
  - DETR
  - d-fine fine-tuning
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Instanzsegmentierung
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -seg im Dateinamen wählt den Mask-Head, ein
        # task-Argument ist hier also nicht nötig.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Instanzsegmentierung
      language: bash
      code: >
        # Setzt auf publizierten Segmentierungsgewichten auf, Mask-Head
        inklusive.

        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Segmentierung aus Detect-Gewichten
      language: bash
      code: |
        # Detect-Gewichte haben keinen Mask-Head, das ist also ein expliziter
        # Transfer: Der Head startet untrainiert und ist erst nach dem Training
        # nützlich. task=segment autorisiert den Transfer hier.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Instanzsegmentierung
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # Masken
        print(metrics["metrics/mAP50-95(B)"])   # Boxen
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über das Dateisuffix, ein exportiertes Artefakt
        # lädt also wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## Installation

D-FINE braucht kein optionales Extra. Alles, was es importiert, steckt in der
Basisinstallation.

```bash
pip install libreyolo
```

Fine-Tuning per Adapter mit `lora=True` ist die Ausnahme und braucht das Extra
`lora`.

```bash
pip install "libreyolo[lora]"
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt ist dasselbe, das jede Familie liefert, ein
Wechsel auf einen anderen Detektor ist also eine Sache von einer Zeile. Ein
Dateiname mit `-seg` löst von sich aus auf die Segmentierungsaufgabe auf, und
`result.masks` trägt dann die Instanzmasken neben den Boxen. `conf` und
`max_det` filtern die Auswahl der Queries; `iou` wird aus API-Parität
akzeptiert, hat aber keine Wirkung, weil der Decoder ein Set Predictor ohne
NMS-Schritt ist. Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und
die Behandlung der Ergebnisse.

## Varianten

Fünf Größen. Sie laufen alle mit derselben Eingabeauflösung, die Tabelle trennt
sie deshalb nach Parameterzahl und Accuracy.

<benchmark-table task="detect" />

<va-embed />

Die Segmentierung nutzt Backbone, Encoder und Decoder der Detektion weiter und
ergänzt einen Mask-Head, ein `-seg`-Checkpoint nimmt also dieselben Argumente
wie sein Detect-Pendant. Die RT-DETRv4-Familie von LibreYOLO ist als Unterklasse
des D-FINE-Wrappers geschrieben: Sie erbt diese Decoder-Linie und schränkt ihre
Aufgabenliste dann wieder auf die Detektion ein, weil sie keinen Mask-Head
trägt.

## Training

Das Training startet für beide Aufgaben von einem publizierten Checkpoint.

<code-tabs name="train" />

Ohne Eingriff läuft der Trainer 132 Epochen mit `lr0=2e-4` und `amp=False`,
einem Batch von 16 und Early Stopping nach 50 Epochen ohne Verbesserung.
Detect-Gewichte sind ein legitimer Startpunkt für ein Segmentierungstraining,
aber nur als expliziter Transfer, denn der Mask-Head beginnt untrainiert und
würde sonst bedeutungslose Masken liefern. Autorisiert wird das dadurch, dass du
`task=segment` an die CLI übergibst. Der Python-Weg ist enger: `LibreDFINE` muss
direkt mit `allow_detect_to_segment_transfer=True` konstruiert werden, weil die
Factory `LibreYOLO()` kein solches Argument nimmt, und die direkte Konstruktion
lädt nichts herunter, die Gewichtsdatei muss also schon auf der Platte liegen.

`lora=True` gilt für die Detektion. Das Segment-Training lehnt es ab und
verweist stattdessen auf `freeze='backbone'`, weil der Mask-Head nicht mit
Adaptern getestet ist. Auf Apple Silicon verschiebt der Trainer den kompletten
Lauf auf die CPU: Der Backward-Pass des gebinnten Matmuls des Integral läuft in
einen Metal-Kompilierfehler. Die Inferenz auf MPS ist davon nicht betroffen.

Siehe [Training](/docs/train) für Datensätze, Augmentierung, Multi-GPU und
Logger.

## Validierung

`val()` gibt ein Dictionary zurück, das nach Metriknamen indiziert ist, und gibt
Ergebnisse pro Klasse aus, solange `verbose` aktiv bleibt.

<code-tabs name="val" />

Bei einem `-seg`-Checkpoint trägt der schlichte Schlüssel `metrics/mAP50-95` den
Mask-Score, und derselbe Lauf meldet zusätzlich Boxen unter `(B)` und Masken
unter `(M)`, beides ist also aus einem Durchlauf verfügbar.

## Export

<export-matrix />

Ein exportiertes Artefakt lädt über sein Dateisuffix wieder durch
`LibreYOLO()`, eine `.onnx`- oder `.engine`-Datei verhält sich also wie ein
Checkpoint und liefert dieselben `Results`. Die Wege über OpenVINO, Paddle, MNN
und Core AI exportieren mit fester Canvas-Größe statt mit dynamischen Formen.
[Export](/docs/export) listet die Argumente auf, die jedes Format akzeptiert,
und die Extras, die einige davon ergänzen.

<code-tabs name="export" />

## Checkpoints

Jede publizierte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

Die Segmentierungsgewichte haben einen zweiten Upstream: Ihr Mask-Decoder, ihr
Mask-Matching und ihr Mask-Loss stammen aus ArgoHA/D-FINE-seg, ebenfalls
Apache-2.0, dessen Maintainer die Nachnutzung mit Namensnennung genehmigt hat.

</provenance-box>

## Zitieren

<citation-block />
