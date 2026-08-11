---
title: DEIM
families:
  - deim
seo_title: DEIM und DEIMv2 in LibreYOLO
description: >-
  Nutze DEIM und DEIMv2 in LibreYOLO für die Objekterkennung. Installation,
  Vorhersage, Training, Validierung und Export, ab einer Größe von einer halben
  Million Parametern.
lead: >-
  Ein Detection Transformer, der mit dichtem Eins-zu-eins-Matching trainiert
  wird und dadurch in weit weniger Epochen konvergiert als die DETR-Rezepte, auf
  denen er aufbaut. LibreYOLO führt zwei Versionen davon, unterschieden durch
  den Checkpoint, den du lädst.
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - DETR
  - detection transformer
  - objekterkennung python
  - echtzeit objekterkennung
  - deim mit eigenem datensatz trainieren
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Die Version steckt im Dateinamen, und die Factory routet über den
        # Checkpoint, also laden beide gleich.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Jede Quelle, die die Bibliothek akzeptiert: Datei, Ordner, URL,
        # Webcam-Index, RTSP-Stream oder eine .streams-Liste
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml lädt beim ersten Aufruf 128 Beispielbilder. Für einen
        # echten Lauf zeigt `data` auf dein eigenes Dataset-YAML.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Bleiben epochs, batch, imgsz und lr0 ungesetzt, kommen sie aus dem
        # veröffentlichten Rezept für die geladene Größe.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Braucht das lora-Extra: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() gibt ein einfaches dict zurück, kein Objekt
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Gegen COCO
      language: bash
      code: |
        # coco-val-only.yaml holt die 5000 val2017-Bilder und lässt das
        # Trainingsset aus. Es enthält ein Download-Skript, braucht also
        # explizite Erlaubnis, sofern der Datensatz nicht schon lokal ist.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Braucht das onnx-Extra: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung: Ein exportiertes Artefakt
        # lädt wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## Installation

Keine der beiden Versionen braucht ein optionales Extra. Alles, was sie
importieren, steckt in der Basisinstallation.

```bash
pip install libreyolo
```

Fine-Tuning mit Adaptern über `lora=True` ist die Ausnahme und braucht das
`lora`-Extra.

```bash
pip install "libreyolo[lora]"
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt ist dasselbe, das jede Familie liefert, ein
anderer Detektor ist also eine Änderung von einer Zeile. `conf` und `max_det`
filtern eine Top-k-Dekodierung über Queries und Klassen; es gibt keinen
NMS-Schritt zum Einstellen, und `iou` wird akzeptiert, aber nicht genutzt.
Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und den Umgang mit
Ergebnissen.

## Varianten

Version 1 bringt fünf Größen mit, alle bei derselben Eingabegröße. Version 2
behält diese fünf Namen und ergänzt drei kleinere, `atto`, `femto` und `pico`,
von denen die ersten beiden nativ auf einer kleineren Eingabegröße laufen als
der Rest. Fünf Größenkürzel gibt es damit in beiden Versionen, und sie benennen
unterschiedliche Modelle; die Version steht im Dateinamen des Checkpoints.

<benchmark-table task="detect" />

<va-embed />

Version 1 behält die Architektur von D-FINE und tauscht deren
Klassifikationsziel gegen den matchability-aware Loss aus dem dichten
Eins-zu-eins-Rezept, deshalb teilen sich die beiden Familien fast jeden
State-Dict-Key und werden über die Metadaten im Checkpoint unterschieden.
Version 2 behält diesen Trainingsvertrag bei und mischt die Backbones: HGNetv2
unterhalb von `s`, und ein DINOv3-Vision-Transformer mit einem
Spatial-Tuning-Adapter ab `s` aufwärts. Dieses Backbone ist der Grund, warum auf
diesen vier Checkpoints eine zweite Lizenz liegt, lies also
[Lizenzierung](#licensing), bevor du einen davon ausrollst.

## Training

Das Training startet von einem veröffentlichten Checkpoint. `pretrained`
erreicht den Trainer nie: Version 1 warnt, dass der Key unbekannt ist, und
ignoriert ihn, Version 2 entfernt ihn. Keine der beiden gibt dir ein zufällig
initialisiertes Modell.

<code-tabs name="train" />

Übergib `lr0` bei Version 1 selbst. Ihre Python-Signatur von `train()` hat
`4e-4` als Standard, die Rate aus dem veröffentlichten COCO-Rezept, während die
Trainingskonfiguration der Familie `1e-4` als Fine-Tuning-Standard führt, und
diesen niedrigeren Wert löst die CLI auf, wenn das Argument fehlt. Die
Konfiguration hält die Messung dahinter fest: bei den Batch-Größen, die ein
Fine-Tuning tatsächlich nutzt, und auf kleinen Datensätzen hat die COCO-Rate
den Transfer messbar verschlechtert.

Version 2 löst diese Standardwerte selbst auf. Lässt du `epochs`, `batch`,
`imgsz` und `lr0` ungesetzt, liest sie jeden Wert aus dem veröffentlichten
Rezept für die geladene Größe, dadurch trainieren die kleinen Größen ohne
weitere Angabe in ihrer eigenen Eingabeauflösung, und ein Wert, den du
übergibst, überschreibt das Rezept. `imgsz` ist das Argument, das sie
einschränkt: Es muss ein positives Vielfaches von 32 sein, sonst wirft
Version 2 einen Fehler, bevor der Lauf startet.

Siehe [Training](/docs/train) für Datensätze, Augmentierung, Multi-GPU und
Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Keys zurück, das Precision, Recall,
mAP 50 und mAP 50-95 abdeckt, gemessen an jedem Datensatz im Format, auf dem du
trainiert hast.

<code-tabs name="val" />

Die Zeilen in der Benchmark-Tabelle oben stammen aus dem Benchmark-Harness von
LibreYOLO; die Notiz unter dieser Tabelle hält fest, welcher Datensatz sie
erzeugt hat, und verlinkt die Laufprotokolle.

## Export

<export-matrix />

Die Matrix deckt beide Versionen auf einer Seite ab: Wo sie sich bei einem
Format unterscheiden, zeigt die Zelle den schwächeren der beiden Werte, hier
wird also für keine der Versionen zu viel versprochen.

Ein exportiertes Artefakt lädt über seine Dateiendung wieder durch
`LibreYOLO()`, eine `.onnx`- oder `.engine`-Datei verhält sich also wie ein
Checkpoint und liefert dieselben `Results`.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>
Die vier DEIMv2-Größen ab S aufwärts beziehen ihr Backbone von DINOv3, deshalb
tragen ihre Gewichts-Repositories sowohl Apache-2.0 als auch Metas DINOv3
License, und LibreYOLO liefert den Quellcode des DINOv3-Backbones unter
derselben Vereinbarung aus. Der Rest dieser Familie, einschließlich jeder
DEIMv2-Größe unterhalb von S, steht allein unter Apache-2.0.
</provenance-box>

## Zitieren

<citation-block />

DEIMv2 ist ein eigenes Paper und hat einen eigenen Zitierblock unter
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation);
zitiere diesen, wenn du einen Checkpoint von Version 2 verwendet hast.
