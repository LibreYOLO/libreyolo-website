---
title: Objekterkennung
seo_title: Objekterkennung in LibreYOLO
description: >-
  Erkenne Objekte in LibreYOLO als achsenparallele Boxen: unterstützte Familien,
  Labelformat sowie Aufrufe für Vorhersage, Training, Validierung und Export.
lead: >-
  Die Objekterkennung lokalisiert jede Objektinstanz in einem Bild und gibt
  dafür ein achsenparalleles Rechteck, ein Klassenlabel und eine Bewertung
  zurück. Der Aufgabenschlüssel lautet detect.
keywords:
  - Objekterkennung Python
  - Objekte im Bild erkennen
  - Begrenzungsrahmen Erkennung
  - MIT Objekterkennungsbibliothek
  - YOLO Alternative
  - Objektdetektor trainieren
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Andere Familie, gleicher Aufruf'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand des Checkpoints weiter. Jeder Detektor gibt
        dasselbe

        # Results-Objekt zurück, daher erfordert der Familienwechsel nur eine
        Codezeile.

        model = LibreYOLO("LibreDFINEn.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy.shape)
    - label: Video und Streams
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Jede von der Bibliothek akzeptierte Quelle: Datei, Ordner, URL,
        Webcam-Index,

        # RTSP-Stream oder eine .streams-Liste.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco128.yaml lädt bei der ersten Verwendung ein Sample mit 128 Bildern
        herunter.

        # Verweise für einen echten Lauf auf die YAML-Datei deines eigenen
        Datensatzes.

        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() gibt ein einfaches Dictionary und kein Objekt zurück.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter. Ein exportiertes
        Artefakt

        # wird wie ein Checkpoint geladen und gibt dasselbe Results-Objekt
        zurück.

        model = LibreYOLO("LibreYOLO9t.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Definition

Die Objekterkennung beantwortet, wo sich jedes Objekt befindet und worum es sich handelt. Ein Bild geht hinein, je Instanz kommt eine Zeile heraus: vier Zahlen für das Rechteck, ein Klassenindex und eine Bewertung. Informationen über Pixelform, Orientierung oder Körperteile sind nicht enthalten. Dadurch unterscheidet sich die Aufgabe von [Instanzsegmentierung](/docs/tasks/instance-segmentation), [orientierten Boxen](/docs/tasks/oriented-detection) und [Posenschätzung](/docs/tasks/pose-estimation).

`detect` ist der kanonische Aufgabenschlüssel und der Standard. Ein Checkpoint ohne Aufgabensuffix im Dateinamen wird als Detektor geladen.

`predict()` befüllt `result.boxes`. `.xyxy` enthält die Pixelecken auf dem ursprünglichen Bild-Canvas, `.conf` die Bewertung und `.cls` den Klassenindex in `result.names`. `.xywh`, `.xyxyn` und `.xywhn` sind abgeleitete Ansichten derselben Zeilen. `.id` enthält eine Tracking-ID, sobald ein Tracker verbunden ist. Das Iterieren über ein `Boxes`-Objekt liefert Ausschnitte mit je einer Zeile. `box.cls`, `box.conf` und `box.xyxy` funktionieren daher für jede einzelne Erkennung.

## Modelle

Zwölf Familien unterstützen sowohl Training als auch Vorhersage: [YOLOv9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter), [RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine), [DEIM](/docs/models/deim), [Dome-DETR](/docs/models/dome-detr), [YOLO-NAS](/docs/models/yolo-nas), [YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7), [RTMDet](/docs/models/rtmdet) und [PicoDet](/docs/models/picodet). YOLOv9 und RF-DETR sind die beiden führenden Familien und erhalten neue Funktionen zuerst. RF-DETR benötigt das eigene Extra `pip install "libreyolo[rfdetr]"`. Die übrigen laufen mit dem Basispaket.

Elf weitere Familien unterstützen Vorhersage, Validierung und Export, während `train()` `NotImplementedError` auslöst: [LW-DETR](/docs/models/lw-detr), [DETR](/docs/models/detr), [Deformable DETR](/docs/models/deformable-detr), [DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn), [Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos), [RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd), [CenterNet](/docs/models/centernet) und [EfficientDet](/docs/models/efficientdet).

Die Darknet-Abstammungslinie aus [YOLOv1](/docs/models/yolov1), [YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) und [YOLOv4](/docs/models/yolov4) wird als eingefrorenes Exponat bewahrt. Vorhersage, Validierung und Export funktionieren, Training nicht.

Eine separate Gruppe übernimmt ihre Klassenliste zur Laufzeit statt aus dem Checkpoint und erkennt dadurch Namen, die beim Training nie vorkamen: [Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2), [OMDet-Turbo](/docs/models/omdet-turbo) und [OV-DEIM](/docs/models/ov-deim) sowie die Vision-Language-Familien [Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2), [Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2), [InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl), [LocateAnything](/docs/models/locate-anything), [SenseNova-Vision](/docs/models/sensenova-vision) und [LibreMODUS](/docs/models/libremodus). Diese Modelle werden über eigene Factorys und Extras geladen. Den genauen Aufruf findest du auf der jeweiligen Modellseite.

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

`conf` legt den Konfidenzschwellenwert fest und `max_det` begrenzt die Zeilenanzahl. `iou` ist der NMS-Schwellenwert und wirkt sich daher nur auf eine Familie aus, die NMS verwendet. RF-DETR und der End-to-End-Kopf von YOLOv9 decodieren eine feste Menge von Vorhersagen und ignorieren ihn. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Datensatzformat

Für jedes Bild gibt es eine `.txt`-Labeldatei. Ihr Pfad entsteht aus dem Bildpfad, indem `images` durch `labels` und die Dateiendung ersetzt wird.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Jede Zeile enthält exakt fünf Felder: einen Klassenindex, gefolgt von einer normalisierten Box aus Mittelpunkt und Größe.

```text
<class_id> <cx> <cy> <w> <h>
```

Koordinaten sind Gleitkommazahlen in `[0, 1]` relativ zur Breite und Höhe des Originalbildes. `w` und `h` müssen positiv sein. Eine fehlende oder leere Labeldatei bedeutet, dass das Bild keine Objekte enthält. Zeilen enthalten weder Konfidenz noch Tracking-ID.

Die YAML-Datei benennt Splits und Klassen:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` und `val` können Bildverzeichnisse, `.txt`-Dateien mit Bildlisten oder Listen aus beidem sein. `nc` ist optional und muss bei Angabe mit `names` übereinstimmen. Natives COCO-JSON wird ebenfalls unterstützt. Füge eine Zuordnung `annotations` von Splitnamen zu JSON-Dateien hinzu. Der Splitpfad gibt dann das Bildstammverzeichnis an. Wenn `names` vorhanden ist, definiert es die Label-IDs. Die Kategorienamen im JSON müssen damit übereinstimmen.

## Training

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` und `lr0` sind die zuerst anzupassenden Argumente. `lr0` lässt sich nicht zwischen Familien übertragen. Eine Lernrate, die ein konvolutionaler Detektor verträgt, kann bei einem Transformer zur Divergenz führen. Übernimm den Wert daher von der Modellseite statt aus dem Beispiel einer anderen Familie. Eine Familie kann ein Argument auch vollständig ignorieren. Die jeweilige Seite führt solche Fälle auf. Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Augmentation, Multi-GPU und Loggern.

## Validierung

`val()` gibt ein einfaches Dictionary mit `metrics/`-Schlüsseln zurück. Diese werden anhand der COCO-Auswertung über den in der Datensatz-YAML-Datei unter `val` benannten Split berechnet.

<code-tabs name="val" />

`metrics/mAP50-95` ist die über IoU-Schwellenwerte von 0,50 bis 0,95 gemittelte Mean Average Precision und die wichtigste Kennzahl. `metrics/mAP50` und `metrics/mAP75` sind die Varianten bei einem einzelnen Schwellenwert. `metrics/mAP_small`, `metrics/mAP_medium` und `metrics/mAP_large` teilen denselben Mittelwert nach Objektfläche auf. `metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium` und `metrics/AR_large` sind die entsprechenden Average-Recall-Werte. `metrics/AR_max_det` und `metrics/max_det` zeichnen die im Lauf verwendete Erkennungsobergrenze auf.

Interpretiere `metrics/precision` und `metrics/recall` bei dieser Aufgabe sorgfältig. Sie bleiben aus Gründen der Abwärtskompatibilität erhalten und sind Aliasse, kein Arbeitspunkt. `metrics/precision` enthält denselben Wert wie `metrics/mAP50-95`, `metrics/recall` denselben wie `metrics/AR100`. Ihre Darstellung als Precision-Recall-Paar gibt einen Wert zweimal wieder. Vier Schlüssel erscheinen außerdem mit dem Suffix `(B)` für Box, damit ein Erkennungsschlüssel bei einem Modell mit zusätzlichen Masken gleich lautet: `metrics/mAP50-95(B)`, `metrics/mAP50(B)`, `metrics/precision(B)` und `metrics/recall(B)`.

## Export

<code-tabs name="export" />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Die Formatabdeckung unterscheidet sich nach Familie. Die Matrix auf jeder Modellseite wird aus der validierten Menge generiert statt manuell geschrieben. Unter [Export und Deployment](/docs/export) findest du Formate, Extras und Einschränkungen.
