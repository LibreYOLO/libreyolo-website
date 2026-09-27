---
title: RT-DETR
families:
  - rtdetr
seo_title: 'RT-DETR, RT-DETRv2 und RT-DETRv4 in LibreYOLO'
description: >-
  Nutze RT-DETR, RT-DETRv2 und RT-DETRv4 in LibreYOLO für Objekterkennung sowie
  orientierte Boxen mit RT-DETRv2. Installiere, sage vorher, trainiere,
  validiere und exportiere mit Gewichten unter Apache-2.0.
lead: >-
  Ein Detection Transformer für Echtzeitinferenz: Er dekodiert statt eines
  dichten Rasters eine feste Menge von Queries und führt daher keine NMS aus.
  LibreYOLO bietet drei Versionen, die durch den geladenen Checkpoint
  unterschieden werden. Version 2 unterstützt außerdem orientierte Boxen.
keywords:
  - rt-detr
  - rt-detrv2
  - rt-detrv4
  - echtzeit detection transformer
  - detr objekterkennung
  - orientierte bounding boxes
  - obb
  - dota
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Die Version ist Teil des Dateinamens, und die Factory richtet sich
        nach dem

        # Checkpoint. Daher werden alle drei gleich geladen.

        model = LibreYOLO("LibreRTDETRv4s.pt")


        # Jede von der Bibliothek unterstützte Quelle: Datei, Ordner, URL,
        Webcam-Index,

        # RTSP-Stream oder eine .streams-Liste

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Orientierte Boxen
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Nur Version 2. Das Suffix -obb wählt die Aufgabe. Der Checkpoint wird
        anhand

        # seiner Tensoren als orientiert erkannt, daher ist kein task-Argument
        nötig.

        # Diese Gewichte nutzen DOTA v1.0 mit 15 Luftbildklassen bei 1024 px.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)     # (N, 5): cx, cy, w, h, Bogenmaß

        print(obb.xyxyxyxy)  # dieselben Zeilen als vier Eckpunkte

        print(result.boxes.xyxy)  # umschließende achsenparallele Boxen
    - label: 'Orientierte Boxen, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRTDETRr18.pt")


        # coco128.yaml lädt bei der ersten Verwendung 128 Beispielbilder
        herunter.

        # Verweise für einen echten Lauf mit `data` auf deine eigene
        Datensatz-YAML.

        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Benötigt das lora-Zusatzpaket: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() gibt ein einfaches dict zurück, kein Objekt
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Gegen COCO
      language: bash
      code: |
        # coco-val-only.yaml ruft die 5000 val2017-Bilder ab und überspringt den
        # Trainingssatz. Sie enthält ein Downloadskript und benötigt daher eine
        # ausdrückliche Erlaubnis, wenn der Datensatz nicht bereits lokal ist.
        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Orientierte Boxen
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Die orientierte Validierung gleicht mit gedrehter IoU ab. Eine
        Vorhersage

        # am richtigen Ort mit falschem Winkel zählt daher als Fehler.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        metrics = model.val(data="my-obb-dataset.yaml")


        print(metrics["metrics/mAP50-95(OBB)"])

        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Benötigt das onnx-Zusatzpaket: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Orientierte Boxen
      language: bash
      code: >
        # ONNX und TorchScript sind die validierten Ziele der orientierten
        Aufgabe,

        # mit FP32, Batch 1 und einer festen Arbeitsfläche von 1024 mal 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreRTDETRr18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## Installation

RT-DETR benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten. Das Zusatzpaket `rtdetr` ist ein stabiler Name,
der nichts ergänzt.

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

Das zurückgegebene `Results`-Objekt entspricht dem aller anderen Familien. Der
Wechsel zu einem anderen Detektor erfordert daher nur eine Änderung in einer
Zeile. `conf` und `max_det` filtern eine Top-k-Dekodierung über Queries und
Klassen. Es gibt keinen NMS-Schritt zum Abstimmen, und `iou` wird zwar
akzeptiert, aber nicht verwendet. Ein orientierter Checkpoint füllt nativ
`result.obb` und außerdem `result.boxes` mit den umschließenden
achsenparallelen Rechtecken. Unter [Vorhersage](/docs/predict) findest du
Quellen, Streaming und die Verarbeitung von Ergebnissen.

## Varianten

Es gibt drei Versionen und insgesamt zwei Aufgaben. Die Größencodes bilden
keine einzige Reihe. Version 1 benennt ihre Größen nach dem Backbone, ResNet
oder HGNetv2. Version 2 verwendet nur die ResNet-Namen erneut. Version 1 bietet
bereits die beiden HGNetv2-Größen, und die Ergebnisse von Version 2 waren dort
so ähnlich, dass LibreYOLO keine doppelten Gewichte veröffentlicht. Version 4
verwendet eine einfache Buchstabenreihe, die mit den HGNetv2-Namen aus Version
1 kollidiert. Ein Größencode allein identifiziert daher kein Modell. Die
Version steht im Dateinamen des Checkpoints.

<benchmark-table task="detect" />

<va-embed />

Version 2 behält Architektur und State-Dictionary-Aufbau von Version 1 bei und
ändert die Abtastung der Deformable Attention. Deshalb unterscheiden die
Metadaten im Checkpoint beide statt der Tensorform. Version 4 hat eine andere
Abstammung: Sie verwendet Architektur und Trainer von D-FINE. Ihre Gewichte
entstehen durch Destillation eines DINOv3-Vision-Foundation-Lehrers in einen
HGNetv2-Schüler. In LibreYOLO ist `LibreRTDETRv4` eine Unterklasse von
`LibreDFINE` mit dauerhaft deaktiviertem Masken-Head und bleibt reine
Objekterkennung.

### Orientierte Boxen mit Version 2

Version 2 ist die einzige Version mit einer zweiten Aufgabe. Ihre unterstützten
Aufgaben sind `detect` und `obb`. Beide verwenden weder denselben Graphen noch
dieselbe Größenreihe. Die Erkennung nutzt ResNet-Größen bei 640 px. Die
orientierte Erkennung nutzt eine HGNetv2-Reihe n, s, m, l und x bei 1024 px.
Die Eingabegröße wird pro Aufgabe statt pro Familie aufgelöst. Ein Checkpoint
wird anhand seiner eigenen Tensoren als orientiert erkannt, nämlich durch die
Box-Heads mit fünf Koordinaten und die Sampling-Parameter von Version 2. Daher
werden `-obb`-Gewichte ohne Argument `task` in den orientierten Graphen geladen.
Eine Abweichung zwischen beiden ist ein eindeutiger Fehler statt einer
unbemerkten Neuinterpretation.

Die veröffentlichten Dateien reichen von `LibreRTDETRv2n-obb.pt` bis
`LibreRTDETRv2x-obb.pt`. Es sind die offiziellen Single-Scale-Checkpoints von
DOTA v1.0, die in das LibreYOLO-Format konvertiert wurden. Sie umfassen 15
Luftbildklassen von Flugzeug und Schiff bis Hafen und Hubschrauber. Die
Klassennamen stehen im Checkpoint. Anders als die Erkennungsseite ist die
orientierte Aufgabe reine Inferenz. Vorhersage, Validierung und Export
funktionieren, `train()` löst bei einem orientierten Modell einen Fehler aus.
Tracking und Test-Time Augmentation unterstützen orientierte Boxen ebenfalls
nicht. [Orientierte Erkennung](/docs/tasks/oriented-detection) behandelt die
Aufgabe, das Labelformat und die Metriken.

## Training

Das Training beginnt mit einem veröffentlichten Checkpoint. `pretrained` wird
bei allen drei Versionen akzeptiert und anschließend verworfen.
`pretrained=False` erzeugt daher kein zufällig initialisiertes Modell. Dieser
Abschnitt betrifft vollständig die Objekterkennung. Die orientierte Aufgabe
von Version 2 ist reine Inferenz. Es gibt keinen Transferpfad von
Erkennungsgewichten, weil beide Aufgaben unterschiedliche Backbones nutzen.

<code-tabs name="train" />

Die Lernrate muss stimmen, und jede Version besitzt einen eigenen Standardwert
statt des bibliotheksweiten Werts. Die Python-Signatur `train()` liest ihn aus
der Trainingskonfiguration der Version. Die CLI löst denselben Wert auf, wenn
`lr0` nicht übergeben wird. Version 1 und 2 akzeptieren außerdem `lr_backbone`
mit standardmäßig einem Zwanzigstel von `lr0`, entsprechend dem ursprünglichen
Rezept. Version 4 verwendet den D-FINE-Trainer, der die
Backbone-Parametergruppe stattdessen mit `backbone_lr_mult` skaliert.

Behalte für `imgsz` die native Größe des Checkpoints bei, sofern kein Grund für
eine Änderung besteht. Validierung und Vorhersage funktionieren mit anderen
Größen, mit einer Ausnahme: Eine rechteckige Größe mit derselben Token-Anzahl
wie die native Größe verwendet weiterhin ein Embedding für das falsche
Seitenverhältnis.

Unter [Training](/docs/train) findest du Datensätze, Datenaugmentierung,
Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln für Precision, Recall,
mAP 50 und mAP 50-95 zurück. Diese werden auf einem beliebigen Datensatz in
dem Format gemessen, das du für das Training verwendet hast.

<code-tabs name="val" />

Die Zeilen der obigen Benchmark-Tabelle stammen aus dem
LibreYOLO-Benchmark-Testaufbau. Der Hinweis unter der Tabelle nennt den
verwendeten Datensatz und verlinkt die Laufaufzeichnungen.

Die orientierte Validierung wird über denselben Aufruf ausgeführt und meldet
dieselben Schlüssel sowie vier Wiederholungen mit dem Suffix `(OBB)`. Der
Abgleich nutzt gedrehte IoU statt der IoU der umschließenden Rechtecke. Ein
Winkelfehler ist daher ein Fehler. `augment=True` wird für diese Aufgabe
abgelehnt.

## Export

<export-matrix />

Die Matrix behandelt die Abstammung auf einer Seite. Wenn sich die drei
Versionen bei einem Format unterscheiden, zeigt die Zelle den schwächsten
Status. Dadurch wird unabhängig von der geladenen Version nichts zu stark
versprochen. Die orientierte Zeile gilt nur für Version 2. ONNX und TorchScript
sind dort mit FP32, Batch 1 und einer festen Arbeitsfläche von 1024 mal 1024
validiert. OpenVINO, TensorRT und ExecuTorch lassen sich konvertieren und neu
laden, haben aber keine Rohdatenparität über die vollständige Query-Menge
erreicht. Die besten Boxen stimmen bis auf einen Bruchteil eines Pixels überein,
während die Ausläufer abweichen.

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

Der Dateiname enthält Version, dann Größe und dann Aufgabe. Erkennungsgewichte
heißen `LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt` und
`LibreRTDETRv4<size>.pt`, alle bei 640 px. Orientierte Gewichte gibt es nur für
Version 2. Sie tragen zusätzlich das Aufgabensuffix und reichen von
`LibreRTDETRv2n-obb.pt` bis `LibreRTDETRv2x-obb.pt`. Alle arbeiten bei 1024 px
und wurden auf DOTA v1.0 statt COCO trainiert.

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />

Der obige Block entspricht der von den Autoren veröffentlichten Zitierangabe
für die Objekterkennung mit Version 1 und 2. Die orientierten Gewichte von
Version 2 haben mit dem Apache-2.0-Repository RiO-DETR einen dritten Upstream:
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR). Von
dort stammen die DOTA-Checkpoints. Zitiere dieses Projekt, wenn du einen davon
verwendet hast. Version 4 ist eine eigene Veröffentlichung einer anderen
Gruppe und besitzt einen eigenen Zitationsblock unter
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation).
Zitiere diesen, wenn du einen Checkpoint der Version 4 verwendet hast.
