---
title: Orientierte Objekterkennung
seo_title: Orientierte Objekterkennung in LibreYOLO
description: >-
  Erkenne gedrehte Objekte in LibreYOLO: Familien für orientierte Boxen,
  Labelzeile mit vier Ecken sowie Aufrufe für Vorhersage, Training, Validierung
  und Export.
lead: >-
  Die orientierte Objekterkennung lokalisiert jede Instanz mit einem gedrehten
  statt einem achsenparallelen Rechteck. Ein geneigtes Objekt wird dadurch eng
  und ohne eine Box voller Hintergrund begrenzt. Der Aufgabenschlüssel lautet
  obb.
keywords:
  - orientierte Begrenzungsrahmen Erkennung
  - gedrehte Objekterkennung
  - OBB Python
  - DOTA Datensatz
  - Objekterkennung Luftbilder
  - Rotated IoU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        # Benötigt das rfdetr-Extra: pip install "libreyolo[rfdetr]"

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Das Suffix -obb im Dateinamen wählt die Aufgabe aus,

        # daher ist kein task-Argument erforderlich.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        result = model(SAMPLE_IMAGE, save=True)


        obb = result.obb

        print(obb.xywhr)   # (N, 5): Mittelpunkt x, Mittelpunkt y, Breite, Höhe,
        Bogenmaß

        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Ecken statt Winkel
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # (N, 4, 2) Eckpunkte in Pixeln
        print(obb.xyxyxyxyn.shape)   # dieselben normalisiert
        print(obb.xyxy.shape)        # (N, 4) umschließende achsenparallele Box
    - label: Kleinerer Checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: >
        from libreyolo import LibreYOLO


        # DOTA-v1.0-Gewichte mit 15 Luftbildklassen bei 1024 px. Der orientierte
        Graph

        # wird anhand der Tensoren des Checkpoints erkannt, daher ist kein
        task-Argument nötig.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)

        print(result.names)   # plane, ship, harbor, helicopter und 11 weitere
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Setzt veröffentlichte orientierte Gewichte fort. data muss auf einen

        # Datensatz mit vier Ecken je Labelzeile verweisen.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: Aus Erkennungsgewichten
      language: bash
      code: >
        # Erkennungsgewichte sagen keinen Winkel vorher, daher ist dies eine
        ausdrückliche

        # Übertragung. Die Angabe task=obb autorisiert sie.

        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() gibt ein einfaches Dictionary und kein Objekt zurück.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNX und TorchScript sind hier die validierten Ziele bei FP32,

        # Batchgröße 1 und einem festen Canvas von 1024 mal 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter. Ein exportiertes
        Artefakt

        # wird wie ein Checkpoint geladen und gibt dasselbe Results-Objekt
        zurück.

        model = LibreYOLO("LibreRFDETRs-obb.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## Definition

Die orientierte Objekterkennung ergänzt eine Erkennung um eine Zahl: den Winkel. Jede Instanz erhält ein gedrehtes Rechteck, eine Klasse und eine Bewertung. Der Vorteil ist die enge Begrenzung. Bei einem Schiff im Winkel von 45 Grad, einem Hallendach oder einer Reihe geparkter Lastwagen enthält eine achsenparallele Box größtenteils Hintergrund. Benachbarte Boxen überlappen sich, obwohl die Objekte es nicht tun. Deshalb ist die Aufgabe bei Luftbildern und Dokumentlayouts üblich und DOTA ihr Referenzdatensatz.

`obb` ist der kanonische Aufgabenschlüssel. Das Suffix `-obb` im Dateinamen eines Checkpoints wählt ihn aus, weshalb beim Laden veröffentlichter Gewichte kein `task=` erforderlich ist.

`predict()` befüllt `result.obb`. `.xywhr` ist die kanonische Form `(N, 5)`: Mittelpunkt x, Mittelpunkt y, Breite, Höhe und ein Winkel im Bogenmaß für die Drehung der Breitseite um den Mittelpunkt. `.conf` und `.cls` enthalten Bewertung und Klassenindex in `result.names`, `.id` beim Tracking eine Tracking-ID. `.xyxyxyxy` wandelt jede Zeile in vier Eckpunkte als Pixel der Form `(N, 4, 2)` um, `.xyxyxyxyn` normalisiert diese Ecken und `.xyxy` liefert die umschließende achsenparallele Box. Verwende Letztere, wenn nachgelagerter Code nur Rechtecke versteht. `result.boxes` wird ebenfalls mit der achsenparallelen Form befüllt.

## Modelle

Zwei Familien unterstützen diese Aufgabe. Die Auswahl hängt davon ab, ob du trainieren möchtest.

[RF-DETR](/docs/models/rf-detr) unterstützt Training. Es sagt orientierte Boxen vorher, trainiert, validiert und exportiert sie und stellt veröffentlichte orientierte Checkpoints in vier Größen bereit: n, s, m und l. Die Familie benötigt das eigene Extra `pip install "libreyolo[rfdetr]"`. Lizenz und Herkunft der Gewichte stehen auf ihrer Modellseite.

Lies vor der Planung mit diesen Checkpoints den folgenden Abschnitt über ihre tatsächlichen Vorhersagen.

[RT-DETRv2](/docs/models/rt-detr) bietet Gewichte für Luftbilder. Die Checkpoints `LibreRTDETRv2n-obb.pt` bis `LibreRTDETRv2x-obb.pt` sind die offiziellen, in das LibreYOLO-Format konvertierten Single-Scale-Checkpoints für DOTA v1.0. Sie decken die 15 DOTA-Klassen bei 1024 px ab. Neben dem Basispaket ist kein Extra erforderlich. Der orientierte Graph wird anhand der Tensoren des Checkpoints erkannt. Vorhersage, Validierung sowie Export nach ONNX und TorchScript werden unterstützt. Training wird nicht unterstützt. Die orientierte Aufgabe ist bei dieser Familie nur für die Inferenz vorgesehen, `train()` löst einen Fehler aus und eine Übertragung aus den Erkennungsgewichten mit anderem Backbone ist nicht möglich. Tracking und Test-Time Augmentation sind für orientierte Boxen ebenfalls nicht verfügbar.

Kurz gesagt: Verwende RT-DETRv2 für fertige DOTA-Kategorien und RF-DETR für eigene orientierte Labels.

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Informiere dich vor der Ausführung über die veröffentlichten RF-DETR-Checkpoints. Obwohl DOTA der Referenzbenchmark dieser Aufgabe ist, wurden diese Gewichte nicht darauf trainiert. Alle vier wurden mit RF-DETR-Erkennungsgewichten initialisiert und auf einem einzigen Roboflow-Universe-Datensatz mit UAV-Aufnahmen feinabgestimmt. Er enthält sechs Fahrzeugklassen: bike, bus, car, other_vehicle, taxi und truck. Die Modellkarten bezeichnen sie als Entwicklungsgewichte, die während der Prüfung der Unterstützung für orientiertes Training entstanden. Sie sollen nicht als produktionsreife oder offizielle Benchmark-Gewichte verstanden werden.

In der Praxis sind sie ein funktionierender Ausgangspunkt für orientierte Boxen um Fahrzeuge aus der Vogelperspektive und zur Prüfung einer vollständigen Pipeline. Jede andere Domäne erfordert Training mit eigenen orientierten Labels. Für die bekannten Luftbildkategorien von DOTA wurden tatsächlich die RT-DETRv2-Checkpoints auf diesen Daten trainiert. `conf` und `max_det` formen die Ausgabe wie bei der Objekterkennung. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Datensatzformat

Die Struktur entspricht dem Erkennungslayout. Für jedes Bild gibt es eine `.txt`-Labeldatei. Ihr Pfad entsteht aus dem Bildpfad, indem `images` durch `labels` und die Dateiendung ersetzt wird.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

Eine Zeile enthält exakt neun Felder: einen Klassenindex, gefolgt von vier geordneten Eckpunkten.

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Die vier Punkte sind normalisierte Gleitkommazahlen in `[0, 1]` und müssen ein nicht degeneriertes orientiertes Rechteck bilden. Die Labeldatei speichert keinen Winkel. Der Loader leitet die kanonische Form `xywhr` aus den Ecken ab. Standardmäßig ist der Parser streng und lehnt Koordinaten außerhalb des Bereichs ab. Bei der Aufnahme für Datensatz und Validierung dürfen anderweitig gültige Labels an Beschnittgrenzen zunächst auf `[0, 1]` begrenzt werden. Degenerierte Boxen werden weiterhin abgelehnt.

Die Zeilenanalyse berücksichtigt die Aufgabe. Neun Felder bezeichnen nur im Modus `obb` eine orientierte Box. Im Modus `segment` wird dieselbe Zeile als Polygon mit vier Punkten gelesen.

Die YAML-Datei entspricht der Erkennungs-YAML:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

Natives COCO-JSON wird mit einer Zuordnung `annotations` von Splitnamen zu JSON-Dateien ebenfalls geladen. Annotationen werden in folgender Priorität gelesen: ein Feld `obb` mit acht Ecken im Pixelraum, ein Feld `obb` als `[cx, cy, w, h, angle]` mit Winkel im Bogenmaß, ein `segmentation`-Polygon oder RLE, das an sein flächenkleinstes Rechteck angepasst wird, oder eine gewöhnliche COCO-`bbox`. Letztere wird als achsenparalleles Rechteck behandelt und zu `xywhr` kanonisiert.

Der kanonische Zeilenparser heißt `libreyolo.data.parse_yolo_obb_label_line`.

## Training

<code-tabs name="train" />

Training für diese Aufgabe bedeutet RF-DETR. Standardmäßig wird das Training aus einem veröffentlichten `-obb`-Checkpoint fortgesetzt. Der Start mit Erkennungsgewichten ist eine bewusste Übertragung. Diese Gewichte sagen keinen Winkel vorher und erst `task=obb` autorisiert den Austausch. Halte `lr0` wie bei den anderen Aufgaben der Familie bei höchstens `1e-4`. Die orientierten Checkpoints von RT-DETRv2 lassen sich nicht feinabstimmen. Verwende sie unverändert oder trainiere ein RF-DETR-Modell mit eigenen Labels. Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Augmentation, Multi-GPU und Loggern.

## Validierung

`val()` gibt ein einfaches Dictionary mit `metrics/`-Schlüsseln zurück. Die Zuordnung verwendet Rotated IoU zwischen orientierten Rechtecken statt zwischen ihren achsenparallelen Hüllen. Eine Vorhersage mit korrekter Position und falschem Winkel wird dadurch als Fehler bewertet.

<code-tabs name="val" />

`metrics/mAP50-95` ist die Mean Average Precision über IoU-Schwellenwerte von 0,50 bis 0,95 in Schritten von 0,05 und die wichtigste Kennzahl. Anders als der COCO-Pfad der Objekterkennung berücksichtigt diese Aufgabe `iou_thresholds` in der Validierungskonfiguration. Der Durchlauf lässt sich daher ändern. `metrics/mAP50` und `metrics/mAP75` sind Varianten bei einem einzelnen Schwellenwert. `metrics/precision` und `metrics/recall` sind echte Precision und Recall bei IoU 0,50 am lockersten Arbeitspunkt. Jede Vorhersage oberhalb des Konfidenzschwellenwerts wird gezählt. Während der Validierung ist dieser standardmäßig 0,001. Eine Erhöhung von `conf` verändert diese Werte, während die über die gesamte Precision-Recall-Kurve berechneten mAP-Werte gleich bleiben. Vier Werte wiederholen sich mit dem Suffix `(OBB)`: `metrics/mAP50-95(OBB)`, `metrics/mAP50(OBB)`, `metrics/precision(OBB)` und `metrics/recall(OBB)`. So kann ein Aufrufer ein orientiertes von einem achsenparallelen Ergebnis in derselben Tabelle unterscheiden. `metrics/mAP75` besitzt keinen Alias mit Suffix.

Zwei Optionen haben bei dieser Aufgabe keine Wirkung. `save_json` und `save_plots` werden akzeptiert und protokollieren eine Warnung. Dumps orientierter Vorhersagen und Validierungsdarstellungen sind nicht implementiert.

## Export

<code-tabs name="export" />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. Die Formatabdeckung unterscheidet sich bei derselben Familie je Aufgabe. Die Matrix auf der Modellseite wird aus der validierten Menge generiert und nennt den Grund, wenn ein Ziel nicht verfügbar ist. Unter [Export und Deployment](/docs/export) findest du Formate, Extras und Einschränkungen.
