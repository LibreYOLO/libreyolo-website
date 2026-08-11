---
title: Instanzsegmentierung
seo_title: Instanzsegmentierung in LibreYOLO
description: >-
  Segmentiere einzelne Objekte in LibreYOLO: die Familien, die den Task
  bedienen, das Polygon-Labelformat und die Aufrufe für Vorhersage, Training,
  Validierung und Export.
lead: >-
  Die Instanzsegmentierung lokalisiert jede Objektinstanz und liefert für jede
  eine Maske pro Pixel, zusätzlich zu Box, Klasse und Score, die ein Detektor
  zurückgibt. Der Task-Key ist segment.
keywords:
  - instanzsegmentierung python
  - objektmasken vorhersagen
  - segmentierungsmodell trainieren
  - polygon labels
  - MIT lizenz segmentierung bibliothek
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -seg im Dateinamen wählt den Masken-Head, ein
        # task-Argument ist also nicht nötig.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), eine Maske pro Detektion
        print(result.boxes.xyxy.shape)   # (N, 4), dieselben N Zeilen
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Maskenkonturen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE)

        # .xy ist eine Liste von (P, 2)-Konturen in Pixeln, .xyn normalisiert.
        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'Andere Familie, gleicher Aufruf'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Setzt auf veröffentlichten Segmentierungsgewichten auf, inklusive

        # Masken-Head. data muss auf Labels mit Polygonen zeigen.

        model = LibreYOLO("LibreDFINEn-seg.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Aus Detektionsgewichten
      language: bash
      code: |
        # Detektionsgewichte tragen keinen Masken-Head, das ist also ein
        # bewusster Transfer: Der Head startet untrainiert. Erlaubt wird
        # das erst, indem du task=segment anforderst.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # Masken
        print(metrics["metrics/mAP50-95(M)"])    # Masken, explizit
        print(metrics["metrics/mAP50-95(B)"])    # Boxen
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung: Ein exportiertes Artefakt
        # lädt wie ein Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Definition

Die Instanzsegmentierung ist Detektion plus Form. Jede Objektinstanz bekommt
weiterhin eine Box, eine Klasse und einen Score, und zusätzlich eine binäre
Maske über den Pixeln, die zu ihr gehören. Masken dürfen sich überlappen, und
Pixel, die zu keinem Objekt gehören, bleiben unzugeordnet, und genau das trennt
den Task von der
[semantischen Segmentierung](/docs/tasks/semantic-segmentation) und der
[panoptischen Segmentierung](/docs/tasks/panoptic-segmentation).

`segment` ist der kanonische Task-Key, und das Suffix `-seg` im Dateinamen eines
Checkpoints wählt ihn aus, `task=` ist also beim Laden veröffentlichter Gewichte
nicht nötig.

`predict()` füllt `result.masks` neben `result.boxes`. `.data` ist ein
`(N, H, W)`-Stapel auf der ursprünglichen Bildfläche, zeilenweise ausgerichtet
an den Boxen, Maske `i` gehört also zu Box `i`. `.xy` wandelt jede Maske in ihre
größte äußere Kontur als `(P, 2)`-Pixel-Array um, und `.xyn` liefert dieselbe
Kontur normalisiert.

## Modelle

Vier Familien trainieren und sagen Masken vorher:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[D-FINE](/docs/models/d-fine) und [RTMDet](/docs/models/rtmdet). RF-DETR braucht
sein eigenes Extra, `pip install "libreyolo[rfdetr]"`; die anderen drei laufen
mit dem Basispaket.

[Mask R-CNN](/docs/models/mask-rcnn) sagt Masken vorher, validiert und
exportiert sie, aber sein `train()` löst `NotImplementedError` aus.

[EoMT](/docs/models/eomt) sagt Masken vorher und validiert sie, kann ebenfalls
nicht trainieren, und sein Export ist noch enger gefasst: `export()` akzeptiert
nur den semantischen Task und löst für `segment` und `panoptic`
`NotImplementedError` aus, weil der Runtime-Vertrag für Query-Masken, den diese
beiden brauchen, nicht definiert ist. Nutze EoMT für Instanzmasken in Python,
nicht über einen exportierten Graph.

Eine eigene Gruppe segmentiert nach einem Prompt statt nach einer Klassenliste:
Ein Klick, eine Box oder eine Phrase wählt das Objekt, und das Modell liefert
seine Maske. [SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) und [PicoSAM3](/docs/models/picosam3)
funktionieren so, ebenso
[SenseNova-Vision](/docs/models/sensenova-vision), dessen Segmentierung
referenzierend ist: Es nimmt eine Phrase, die ein Objekt benennt. Sie laden über
ihre eigene Factory und ihre eigenen Extras, und jede Modellseite trägt den
genauen Aufruf.

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

`conf` und `max_det` formen die Ausgabe genauso wie bei der Detektion, und
Masken werden zusammen mit den Boxen gefiltert, zu denen sie gehören. Siehe
[Vorhersage](/docs/predict) für Quellen, Streaming und den Umgang mit
Ergebnissen.

## Datensatzformat

Das Layout ist das Detektionslayout: eine `.txt`-Labeldatei pro Bild, gefunden
durch Austausch von `images` gegen `labels` im Bildpfad und einen Wechsel der
Dateiendung.

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

Was sich ändert, ist die Zeile. Ein Segment ist ein Klassenindex, gefolgt von
einem flachen Polygon:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Mindestens drei Punkte, die Anzahl der Koordinaten nach dem Klassenindex ist
also gerade und mindestens sechs, und das Polygon darf nicht entartet sein. Die
Koordinaten sind Floats in `[0, 1]` relativ zu Breite und Höhe des
Originalbildes. Eine Detektionszeile mit fünf Feldern wird in einem
Segmentierungsdatensatz ebenfalls akzeptiert und als rechteckiges Segment
gelesen, was einen reinen Box-Datensatz ohne Konvertierungslauf ladbar macht.

Die YAML ist die Detektions-YAML:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

Natives COCO JSON funktioniert ebenfalls: Ergänze eine `annotations`-Zuordnung
von Split-Name zu JSON-Datei, und der Split-Pfad gibt das Bildwurzelverzeichnis
an.

## Training

<code-tabs name="train" />

Das Training setzt standardmäßig auf einem veröffentlichten `-seg`-Checkpoint
auf. Von Detektionsgewichten aus zu starten ist möglich, aber ein bewusster
Transfer: Diese Gewichte tragen keinen Masken-Head, er startet also
untrainiert, und `task=segment` zu übergeben ist das, was den Tausch erlaubt.
Siehe [Training](/docs/train) für Datensätze, Datenaugmentierung, Multi-GPU und
Logger.

## Validierung

`val()` liefert ein schlichtes Dictionary von `metrics/`-Keys. Boxen und Masken
werden getrennt bewertet, beide mit COCO-Evaluation, und die Maskenzahlen sind
die primären.

<code-tabs name="val" />

Die Keys ohne Suffix halten die Maskenergebnisse: `metrics/mAP50-95`,
`metrics/mAP50`, `metrics/mAP75`, dann `metrics/mAP_small`,
`metrics/mAP_medium` und `metrics/mAP_large` nach Objektfläche sowie
`metrics/AR1`, `metrics/AR10`, `metrics/AR100`, `metrics/AR_small`,
`metrics/AR_medium`, `metrics/AR_large` für den durchschnittlichen Recall.
`metrics/AR_max_det` und `metrics/max_det` halten die Detektionsobergrenze fest,
die der Lauf genutzt hat.

Vier Kennzahlen werden zusätzlich unter einem expliziten Suffix
veröffentlicht, `(M)` für Maske und `(B)` für Box, damit ein Vergleich nie davon
abhängt, welche Zahl die Familie als primär bezeichnet hat:
`metrics/mAP50-95(M)` und `metrics/mAP50-95(B)`, `metrics/mAP50(M)` und
`metrics/mAP50(B)`, `metrics/precision(M)` und `metrics/precision(B)`,
`metrics/recall(M)` und `metrics/recall(B)`. Ein `metrics/precision` oder
`metrics/recall` ohne Suffix gibt es bei diesem Task nicht.

Lies die Precision- und Recall-Keys sorgfältig. Sie bleiben aus
Abwärtskompatibilität erhalten und sind Aliase, kein Arbeitspunkt:
`metrics/precision(M)` hält denselben Wert wie `metrics/mAP50-95(M)`, und
`metrics/recall(M)` denselben Wert wie der Masken-AR bei 100 Detektionen, wobei
sich `(B)` für Boxen genauso verhält. Ein Paar davon zu plotten zeigt eine Zahl
zweimal.

## Export

<code-tabs name="export" />

Ein exportiertes Artefakt lädt über `LibreYOLO()` anhand seiner Dateiendung
zurück, deshalb verhält sich eine `.onnx`- oder `.engine`-Datei wie ein
Checkpoint und liefert dieselben `Results`. Die Abdeckung für die Segmentierung
ist enger als die für die Detektion bei derselben Familie. Die Matrix auf jeder
Modellseite wird aus dem validierten Satz erzeugt und nennt den Grund, warum ein
Ziel nicht verfügbar ist. Siehe
[Export und Deployment](/docs/export) für die Formate, ihre Extras und ihre
Einschränkungen.
