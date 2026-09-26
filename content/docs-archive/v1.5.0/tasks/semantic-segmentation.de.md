---
title: Semantische Segmentierung
seo_title: Semantische Segmentierung in LibreYOLO
description: >-
  Weise mit LibreYOLO jedem Pixel eine Klasse zu. Erfahre mehr über die
  unterstützten Familien, das Format dichter Masken sowie die Aufrufe für
  Vorhersage, Training, Validierung und Export.
lead: >-
  Die semantische Segmentierung weist jedem Pixel eines Bilds eine Klasse zu und
  unterscheidet nicht zwischen Instanzen derselben Klasse. Der Aufgabenschlüssel
  lautet semantic.
keywords:
  - semantische segmentierung python
  - pixel klassifikation
  - dichte vorhersage
  - segmentierungsmodell trainieren
  - mIoU
  - MIT segmentation bibliothek
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -sem im Dateinamen wählt die Aufgabe aus. Ein task-
        # Argument ist daher nicht nötig.
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) Klassen-IDs auf der Original-Canvas
        print(mask.classes)      # sortierte Klassen-IDs ohne 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Jeweils eine Klasse
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # boolesch (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Andere Familie, gleicher Aufruf'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Auf ADE20K
      language: bash
      code: >
        # ade20k.yaml enthält ein Downloadskript für das etwa 1 GB große

        # Archiv. Ohne lokale Daten ist daher eine ausdrückliche Erlaubnis
        nötig.

        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() gibt ein einfaches Dictionary und kein Objekt zurück.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet nach Dateiendung. Ein exportiertes Artefakt wird
        # wie ein Checkpoint geladen und gibt dasselbe Results-Objekt zurück.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Definition

Die semantische Segmentierung weist Pixeln und nicht Objekten Labels zu. Jedes
Pixel erhält eine Klassen-ID. Zwei Autos, die sich im Bild berühren, bilden
eine einzige Region der Autoklasse ohne Grenze zwischen ihnen. Das Zählen von
Instanzen ist Aufgabe der
[Instanzsegmentierung](/docs/tasks/instance-segmentation). Wenn jedes Pixel
beschriftet und zugleich jede Instanz getrennt werden soll, ist die
[panoptische Segmentierung](/docs/tasks/panoptic-segmentation) zuständig.

`semantic` ist der kanonische Aufgabenschlüssel. Das Suffix `-sem` im Namen
einer Checkpoint-Datei wählt diese Aufgabe aus. Beim Laden veröffentlichter
Gewichte ist `task=` daher nicht erforderlich.

`predict()` füllt `result.semantic_mask`. `.data` ist eine ganzzahlige
Klassenkarte der Form `(H, W)` auf der Canvas des Originalbilds. `.classes`
führt die vorhandenen IDs in sortierter Reihenfolge auf und `.class_mask(id)`
gibt die boolesche `(H, W)`-Auswahl für eine Klasse zurück. Der Wert `255` ist
das Ignore-Label. Er bildet nie eine Klasse, wird von Loss und Metriken
ausgeschlossen und von `.classes` nicht aufgeführt.

## Modelle

Drei Familien unterstützen sowohl Training als auch Vorhersage:
[SegFormer](/docs/models/segformer),
[LingBot-Vision](/docs/models/lingbot-vision) und
[DINOv2](/docs/models/dinov2). SegFormer und LingBot-Vision laufen mit dem
Basispaket und stellen veröffentlichte Gewichte bereit. DINOv2 benötigt
`pip install "libreyolo[rfdetr]"` und besitzt keinen von LibreYOLO gehosteten
Checkpoint. Es lädt das Upstream-Backbone, während sein dichter Head zufällig
initialisiert wird. Daher ist es ein Ausgangspunkt für das Training und kein
direkt einsetzbarer Prädiktor.

Vier weitere Familien unterstützen Vorhersage, Validierung und Export, aber
ihre Methode `train()` löst `NotImplementedError` aus:
[FCN](/docs/models/fcn), [DeepLabv3](/docs/models/deeplabv3),
[PIDNet](/docs/models/pidnet) und [EoMT](/docs/models/eomt).

Die Klassensätze unterscheiden sich je nach Checkpoint und nicht je nach
Familie. Die veröffentlichten Gewichte stammen aus Datensätzen mit sehr
unterschiedlichen Labelräumen, darunter die 150 Klassen von ADE20K und die 19
Klassen von Cityscapes. Anhand von `names` des Checkpoints erkennst du, welche
Labels er vorhersagen kann. Zwei Checkpoints sind nur vergleichbar, wenn sie
auf demselben Datensatz trainiert wurden.

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert.

<code-tabs name="predict" />

Die Karte entsteht durch ein Argmax pro Pixel. Daher gibt es keinen NMS-Schritt
und `iou` hat nie eine Wirkung. `conf` und `max_det` werden aus Gründen der
API-Parität akzeptiert, haben aber bei SegFormer, PIDNet und den anderen
dichten Prädiktoren keine Wirkung. EoMT bildet die Ausnahme, bei der `conf` die
Query-Auswahl filtert. Unter [Vorhersage](/docs/predict) findest du
Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Datensatzformat

Jedem Bild wird statt einer `.txt`-Labeldatei eine dichte einkanalige Maske
zugeordnet. Sie wird gefunden, indem im Bildpfad `images` durch das
Maskenverzeichnis ersetzt wird.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Masken sind verlustfreie einkanalige Bilder, normalerweise im PNG-Format.
PNG-Dateien im Palettenmodus werden als Palettenindizes gelesen. Jeder
Pixelwert ist eine Klassen-ID im Bereich `0..nc-1`. Der Wert `255` bedeutet
„ignorieren“. Die Auflösung der Maske muss der des zugehörigen Bilds
entsprechen.

Die YAML ergänzt den gemeinsamen Vertrag um zwei Schlüssel:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` bezeichnet den Verzeichnisnamen, der `images` ersetzt. Der
Standardwert lautet `masks`. `label_mapping` ist eine optionale Zuordnung
`{source_id: train_id}`, die beim Laden auf die Pixelwerte der Maske angewendet
wird. So kann ein von 1 bis 150 nummerierter Datensatz in den Bereich 0 bis 149
überführt werden. Jeder nicht zugeordnete Quellwert wird zum Ignore-Label. Jede
Trainings-ID muss im Bereich `0..nc-1` liegen.

Wenn `masks_dir` fehlt, verwendet der Loader einen Fallback: Beim Laden werden
Masken aus Polygonlabels rasterisiert, die mit der üblichen Konvention von
`images` zu `labels` aufgelöst werden. Nach den Objektklassen wird eine Klasse
`background` angehängt, sodass `nc` um eins wächst.

Der kanonische Loader ist `libreyolo.data.SemanticDataset`.

## Training

<code-tabs name="train" />

`imgsz` ist hier stärker eingeschränkt als bei einem Detektor. Jede Familie
deklariert einen Teiler, durch den ihre Eingabegröße ohne Rest teilbar sein
muss. Er wird durch das Patch-Raster oder den Output Stride bestimmt. Wenn
`imgsz` nicht ohne Rest teilbar ist, lösen Training und Validierung vor Beginn
des Laufs einen `ValueError` aus. Der Teiler ist 32 für SegFormer, 16 für
LingBot-Vision und EoMT, 14 für DINOv2 sowie 8 für FCN und PIDNet. Unter
[Training](/docs/train) findest du Informationen zu Datensätzen,
Augmentierung, Multi-GPU und Loggern.

## Validierung

`val()` gibt ein einfaches Dictionary mit `metrics/`-Schlüsseln zurück. Die
Werte werden über den in der Datensatz-YAML als `val` bezeichneten Split
berechnet.

<code-tabs name="val" />

`metrics/mIoU` ist die mittlere Intersection over Union. Für jede Klasse wird
die Überlappung zwischen vorhergesagten und tatsächlichen Pixeln durch ihre
Vereinigung geteilt und anschließend über alle Klassen gemittelt. Dies ist die
wichtigste Kennzahl und wird während des Trainings zur Auswahl der besten
Epoche verwendet. `metrics/pixel_accuracy` ist der Anteil der Pixel mit der
richtigen Klasse. Eine große Hintergrundklasse kann ihn aufblähen, daher
solltest du zum Vergleich die mIoU verwenden. Mit `255` markierte Pixel tragen
zu keinem der beiden Werte bei. Das Dictionary enthält außerdem `fitness` als
Kopie des mIoU-Werts.

## Export

<code-tabs name="export" />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über
`LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich deshalb
wie ein Checkpoint und gibt dieselben `Results` zurück. Die Formatunterstützung
unterscheidet sich je nach Familie. Die Matrix auf jeder Modellseite wird aus
dem validierten Satz generiert und nicht von Hand geschrieben. Unter
[Export und Deployment](/docs/export) findest du die Formate, ihre Extras und
ihre Einschränkungen.

