---
title: Posenschätzung
seo_title: Posenschätzung in LibreYOLO
description: >-
  Sage Keypoints je Instanz in LibreYOLO vorher: unterstützte Familien,
  Labelformat sowie Aufrufe für Vorhersage, Training, Validierung und Export.
lead: >-
  Die Posenschätzung lokalisiert jede Instanz und gibt dafür eine geordnete
  Menge benannter Keypoints zurück. Die Ausgabe enthält dadurch die interne
  Struktur des Objekts statt nur seiner Ausdehnung. Der Aufgabenschlüssel lautet
  pose.
keywords:
  - Posenschätzung Python
  - Keypoint-Erkennung
  - menschliches Pose-Modell
  - COCO Keypoints
  - OKS mAP
  - Pose-Modell trainieren
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -pose im Dateinamen wählt den Keypoint-Kopf aus,
        # daher ist kein task-Argument erforderlich.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) Pixelkoordinaten
        print(result.boxes.xyxy.shape)     # (N, 4), dieselben N Instanzen
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Nur sichtbare Keypoints
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible wird aus der dritten Keypoint-Spalte abgeleitet und
        # ist überall wahr, wenn der Checkpoint nur (x, y) vorhersagt.
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Alternative mit Top-down
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # HRNet arbeitet Top-down und beschneidet zuerst jede Person. Ohne
        angegebene

        # Personenquelle koppelt es sich an LibreYOLO9t und protokolliert diese
        Auswahl.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # coco8-pose.yaml enthält ein Downloadskript und benötigt deshalb

        # eine ausdrückliche Berechtigung, sofern die Daten noch nicht lokal
        vorliegen.

        model = LibreYOLO("LibreECs-pose.pt")

        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Eigener Datensatz
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml muss kpt_shape definieren und die Labelzeilen müssen
        # exakt 5 + K * D Felder enthalten.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreECs-pose.pt")


        # val() gibt ein einfaches Dictionary und kein Objekt zurück.

        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)


        print(metrics["metrics/keypoints_mAP50-95"])

        print(metrics["metrics/keypoints_mAP50"],
        metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter. Ein exportiertes
        Artefakt

        # wird wie ein Checkpoint geladen und gibt dasselbe Results-Objekt
        zurück.

        model = LibreYOLO("LibreECs-pose.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Definition

Die Posenschätzung gibt Struktur statt nur Ausdehnung zurück. Jede Instanz erhält weiterhin eine Box, eine Klasse und eine Bewertung. Zusätzlich erhält sie `K` Keypoints in fester Reihenfolge. Index 5 bezeichnet dadurch bei jeder Instanz und in jedem Bild dasselbe Körperteil. Der Labelsatz definiert diese Reihenfolge. Die Ausgabe selbst benennt keinen Keypoint.

`pose` ist der kanonische Aufgabenschlüssel. Das Suffix `-pose` im Dateinamen eines Checkpoints wählt ihn aus. Beim Laden veröffentlichter Gewichte ist deshalb kein `task=` erforderlich.

`predict()` befüllt neben `result.boxes` auch `result.keypoints`. `.data` besitzt die Form `(N, K, 2)` oder `(N, K, 3)` und ist zeilenweise an den Boxen ausgerichtet. Instanz `i` im einen Objekt entspricht Instanz `i` im anderen. `.xy` wählt die Pixelkoordinaten aus, `.xyn` normalisiert sie anhand der ursprünglichen Bildgröße. `.conf` ist die dritte Spalte, wenn der Checkpoint sie vorhersagt, andernfalls `None`. `.has_visible` ist die daraus abgeleitete boolesche Maske und vollständig wahr, wenn es keine dritte Spalte gibt.

Zwei Architekturarten führen zu dieser Ausgabe. Ein einstufiges Modell sagt Boxen und Keypoints in einem Durchlauf vorher. Ein Top-down-Modell führt zunächst einen Detektor aus, beschneidet jede Instanz und regressiert Keypoints innerhalb des Ausschnitts. Seine Genauigkeit hängt daher vom vorgeschalteten Detektor ab.

## Modelle

Drei einstufige Familien unterstützen Training und Vorhersage: [RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter) und [YOLO-NAS](/docs/models/yolo-nas). RF-DETR benötigt das eigene Extra `pip install "libreyolo[rfdetr]"`. RF-DETR und EdgeCrafter stellen veröffentlichte Pose-Checkpoints bereit und beide lassen sich auf einklassigen Datensätzen mit ausschließlich Personen feinabstimmen. Der Keypoint-Kopf von EdgeCrafter wird bei der Erstellung festgelegt und lehnt Datensätze mit einer anderen Anzahl ab. RF-DETR initialisiert seinen Kopf dafür neu. YOLO-NAS lädt seine Gewichte aus Deci.AIs eigenem CDN unter einer nicht kommerziellen Lizenz. LibreYOLO veröffentlicht keine davon. Sein Pose-Kopf wird ebenfalls für eine neue Keypoint-Anzahl umgebaut. Als einzige der drei Familien ist seine Klassenanzahl nicht auf eins beschränkt. Verwende es daher für ein mehrklassiges oder nicht menschliches Skelett, beispielsweise Tierposen.

[HRNet](/docs/models/hrnet) ist die Top-down-Option. Es unterstützt Vorhersage, Validierung und Export, während `train()` `NotImplementedError` auslöst. Ohne Personenquelle koppelt es sich automatisch an einen LibreYOLO9t-Detektor. `cropped=True` behandelt das gesamte Bild als eine Instanz, `person_boxes=` übernimmt bereits vorhandene Boxen und `person_detector=` benennt einen anderen Detektor.

[SenseNova-Vision](/docs/models/sensenova-vision) gibt ebenfalls Keypoints aus. Es ist ein generatives Modell mit Prompts, eigener Factory `LibreVLM` und eigenem Extra. Ohne festgelegtes Vokabular fällt `set_task("pose")` auf die Kategorie Person zurück. Seine Gewichte dürfen nicht kommerziell verwendet werden. Die Latenz je Bild ist deutlich höher als bei einem eigens entwickelten Pose-Kopf, da jede Vorhersage eine Diffusionsdecodierung ausführt.

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Keypoint-Anzahl und -Reihenfolge sind Eigenschaften des Checkpoints, nicht der Bibliothek. Ein auf einem anderen Skelett trainiertes Modell gibt daher ein anderes `K` und eine andere Bedeutung je Index zurück. Auch der Inhalt der dritten Keypoint-Spalte hängt vom Checkpoint ab. EdgeCrafter schreibt dort statt einer punktweisen Bewertung eine Konstante. Es besitzt überhaupt keinen Boxkopf, weshalb jede Pose-Box die Begrenzung der eigenen Keypoints dieser Instanz darstellt. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Datensatzformat

Die Struktur entspricht dem Erkennungslayout. Für jedes Bild gibt es eine `.txt`-Labeldatei. Ihr Pfad entsteht aus dem Bildpfad, indem `images` durch `labels` und die Dateiendung ersetzt wird.

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

Eine Zeile ist eine Erkennungszeile mit angehängten Keypoints:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Die Feldanzahl beträgt exakt `5 + K * D`, wobei `D` der zweite Wert von `kpt_shape` ist. Box- und Keypoint-Koordinaten sind normalisierte Gleitkommazahlen relativ zur Breite und Höhe des Originalbildes. Der Sichtbarkeitswert `v` ist nur bei `D` gleich 3 vorhanden und lautet `0`, `1` oder `2`.

Die YAML-Datei ergänzt den gemeinsamen Vertrag um zwei Schlüssel:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` ist erforderlich und lautet `[K, 2]` oder `[K, 3]`. `flip_idx` ist optional und eine Permutation von `0..K-1`. Sie gibt für jeden Keypoint den Index nach horizontaler Spiegelung an, damit ein linkes Handgelenk ein linkes Handgelenk bleibt. Wenn du den Wert auslässt, wird die horizontale Spiegelungsaugmentation für Keypoints deaktiviert, statt sie mit falscher Indexreihenfolge anzuwenden.

## Training

<code-tabs name="train" />

Das Training wird mit einem veröffentlichten `-pose`-Checkpoint fortgesetzt, der bereits einen Keypoint-Kopf enthält. Die Aufgabe wird aus dem geladenen Checkpoint gelesen und nicht aus einem zur Trainingszeit übergebenen Flag. Ein Erkennungscheckpoint wird daher nicht durch eine entsprechende Anforderung zu einem Pose-Lauf. `kpt_shape` in deiner YAML-Datei muss bei EdgeCrafter exakt zum Kopf passen, da dieser bei der Erstellung festgelegt wird. RF-DETR und YOLO-NAS ändern den Kopf dagegen für eine andere Anzahl. Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Augmentation, Multi-GPU und Loggern.

## Validierung

`val()` gibt ein einfaches Dictionary mit `metrics/`-Schlüsseln zurück. Die Bewertung verwendet die COCO-Keypoint-Auswertung über Object Keypoint Similarity. Sie gewichtet den Abstandsfehler jedes Keypoints anhand der Instanzgröße und einer Keypoint-bezogenen Toleranz. OKS übernimmt damit die Rolle, die IoU bei Boxen spielt. Benötigt wird `pycocotools`, das in der Basisinstallation enthalten ist.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` ist die wichtigste Kennzahl. Sie entspricht der Mean Average Precision über OKS-Schwellenwerte von 0,50 bis 0,95 und wird vom Training zur Auswahl der besten Epoche verwendet. `metrics/keypoints_mAP50` und `metrics/keypoints_mAP75` sind die Varianten bei einem einzelnen Schwellenwert. `metrics/keypoints_mAP_M` und `metrics/keypoints_mAP_L` teilen den Mittelwert nach mittlerer und großer Instanzfläche auf. Die COCO-Keypoint-Auswertung definiert keine kleine Gruppe. Die entsprechenden Average-Recall-Werte heißen `metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`, `metrics/keypoints_AR75`, `metrics/keypoints_AR_M` und `metrics/keypoints_AR_L`. Jeder Schlüssel dieser Aufgabe trägt das Präfix `keypoints_`. Die von einem Detektor zurückgegebenen Box-`mAP`-Schlüssel erscheinen nicht.

## Export

<code-tabs name="export" />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Die Formatabdeckung unterscheidet sich nach Familie. Die Matrix auf jeder Modellseite wird aus der validierten Menge generiert statt manuell geschrieben. Unter [Export und Deployment](/docs/export) findest du Formate, Extras und Einschränkungen.
