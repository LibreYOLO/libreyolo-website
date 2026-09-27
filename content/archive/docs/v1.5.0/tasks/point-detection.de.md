---
title: Punkterkennung
seo_title: Punkterkennung und Objektzählung in LibreYOLO
description: >-
  Lokalisiere Objekte in LibreYOLO als einzelne Punkte statt als Boxen. Sage
  Mittelpunkte vorher, zähle Objekte, trainiere FOMO und lies die Punktmetriken.
lead: >-
  Die Punkterkennung gibt je Objekt eine Position x, y statt eines
  Begrenzungsrahmens zurück. LibreYOLO stellt sie als Aufgabe point bereit. Eine
  Vorhersage enthält je Objekt eine Zeile aus x, y, Klasse und Konfidenz.
keywords:
  - Punkterkennung Python
  - Objektzählung Python
  - Mittelpunkt Erkennung
  - FOMO Punktlokalisierung
  - Objekte in Bildern zählen
  - Punktlokalisierung
last_verified: 1.5.0
snippets:
  predict:
    - label: Punkte vorhersagen und zählen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # LibreFOMO-Gewichte werden nicht automatisch heruntergeladen. Lade
        zuerst einen Checkpoint

        # von https://huggingface.co/LibreYOLO herunter und öffne ihn über den
        lokalen Pfad.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # Objektanzahl

        print(points.xy)       # (N, 2) Mittelpunkte in Pixeln des
        Originalbildes

        print(points.cls, points.conf)
    - label: Normalisierte Koordinaten und Anzahl je Klasse
      language: python
      code: >
        from collections import Counter


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE)


        points = result.points.numpy()

        print(points.xyn)                          # dieselben Mittelpunkte in
        [0, 1]

        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: FOMO mit einem YOLO-Datensatz trainieren
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Mit dem trainierten Checkpoint vorhersagen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        results = model.train(data="my-dataset.yaml", epochs=40)


        # train() lädt den besten Checkpoint erneut in dasselbe Objekt. Nach
        Abschluss

        # des Aufrufs verwendet das Modell deshalb die trainierten Gewichte.

        print(results["best_checkpoint"])

        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Validieren und Metrikschlüssel lesen
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")

        metrics = model.val(data="my-dataset.yaml")


        print(metrics["metrics/precision"], metrics["metrics/recall"])

        print(metrics["metrics/f1"])

        print(metrics["metrics/mAP@[0.01:0.10]"])   # Fitness

        print(metrics["metrics/MLE"])               # mittlerer
        Lokalisierungsfehler

        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # Zählfehler
    - label: Abstandsschwellenwerte ändern
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Die Durchlaufgrenzen sind Teil des Schlüsseltexts. Ein eigener
        Durchlauf

        # benennt daher die erzeugten mAP-Schlüssel um.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Exportierte Datei ausführen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Definition

Die Aufgabe `point` lokalisiert jedes Objekt ohne Breite, Höhe oder Maske anhand einer einzelnen x-y-Koordinate und einer Klasse. Da eine Vorhersage eine flache Objektliste ist, entspricht die Zeilenanzahl der Objektanzahl. Dadurch wird sie zur Zählaufgabe.

Eine Vorhersage befüllt `result.points`, eine `Points`-Nutzlast um ein Array der Form `(N, 4)` mit Zeilen aus `x, y, class, confidence` in Pixeln des Originalbildes. `.xy` gibt die Koordinaten zurück, `.xyn` dieselben Koordinaten geteilt durch die Bildgröße, `.cls` die Klassenindizes und `.conf` die Bewertungen. `len()` gibt die Punktanzahl zurück. `result.boxes` bleibt leer, weshalb `iou` und `max_det` nichts beeinflussen können.

## Modelle

Drei Familien unterstützen `point` und sind nicht austauschbar.

[FOMO](/docs/models/fomo) ist die Option mit festem Vokabular. Ein Rasterklassifikator markiert jede Zelle eines niedrig aufgelösten Rasters als Hintergrund oder Objektmittelpunkt. Es ist die einzige Punktfamilie, die LibreYOLO trainieren und exportieren kann.

[LocateAnything](/docs/models/locate-anything) verwendet Text statt eines Klassenindex. Das Vokabular entspricht daher jeder angegebenen Formulierung. Die Familie benötigt das Extra `vlm`, wird als `LibreLocateAnything` statt über die Factory `LibreYOLO()` erstellt und ihre Gewichte dürfen nur nicht kommerziell verwendet werden. Die genauen Bedingungen und zwei weiteren im Checkpoint kombinierten Lizenzen stehen auf der Modellseite.

[SenseNova-Vision](/docs/models/sensenova-vision) erreicht `point` über denselben Checkpoint mit Prompt-basierter Generierung, den es für sechs weitere Aufgaben verwendet. Das Modell wird mit `LibreVLM("sensenova-vision", task="point")` geladen und benötigt das Extra `sensenova`. Jede Vorhersage ist ein Generierungsdurchlauf über ein 7B-Modell. Rechne deshalb mit einer deutlich höheren Latenz je Bild als bei einem eigens entwickelten Detektor. Die Gewichte sind nicht kommerziell nutzbar. Die Lizenz steht auf der Modellseite.

## Vorhersage

LibreFOMO-Gewichte bilden auf dieser Website die einzige Ausnahme vom automatischen Download. `LibreYOLO("LibreFOMOs-point.pt")` sucht diese Datei lokal und löst unter Angabe des Namens `ValueError` aus, statt sie abzurufen. Lade zuerst einen Checkpoint von der [LibreYOLO-Organisation](https://huggingface.co/LibreYOLO) auf Hugging Face herunter und öffne ihn über den lokalen Pfad oder trainiere ein eigenes Modell.

<code-tabs name="predict" />

Der Dateiname muss das Aufgabensuffix `-point` enthalten, damit der Loader ihn erkennt. `predict(..., nms_radius=1)` steuert, wie viele Rasterzellen zwei FOMO-Erkennungen voneinander entfernt sein müssen, damit beide erhalten bleiben. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Datensatzformat

`point` besitzt kein eigenes Labelformat. Punktfamilien lesen das gewöhnliche YOLO-Erkennungslayout und leiten aus jeder Boxzeile einen Mittelpunkt ab. `cx cy` ist der Punkt, während `w h` nur die Gültigkeit der Zeile bestimmen.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Jede Labeldatei enthält eine Zeile je Objekt mit normalisierten Koordinaten:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

Eine fehlende oder leere Labeldatei bedeutet, dass keine Objekte vorhanden sind. Den vollständigen Vertrag findest du unter [Datensatzformate](/docs/reference/dataset-formats).

## Training

FOMO ist die einzige Punktfamilie mit einer Trainingsimplementierung. `train()` löst für LocateAnything und SenseNova-Vision `NotImplementedError` aus. Führe das Fine-Tuning dieser Modelle im jeweiligen Upstream-Projekt aus und lade das Ergebnis.

<code-tabs name="train" />

`imgsz` ist bei FOMO nicht frei wählbar. Der Wert entspricht standardmäßig der nativen Auflösung des geladenen Checkpoints. Eine andere Angabe löst `ValueError` unter Nennung der erwarteten Größe aus. Unter [Training](/docs/train) findest du Informationen zu Datensätzen, Loggern und Multi-GPU. Die [FOMO-Seite](/docs/models/fomo) beschreibt die Standardwerte der Familie.

## Validierung

`val()` ordnet vorhergesagte Punkte mit dem ungarischen Algorithmus über einen Durchlauf von Abstandsschwellenwerten eins zu eins den Ground-Truth-Punkten zu. Ein Schwellenwert ist ein euklidischer Abstand in normalisierten Bildkoordinaten. Der Standarddurchlauf enthält zehn Werte von 0,01 bis 0,10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall` und `metrics/f1` werden beim strengsten Schwellenwert des Durchlaufs als Makromittel über die Klassen berechnet, standardmäßig 0,01. `metrics/mAP@0.01` ist die Average Precision bei demselben Schwellenwert. `metrics/mAP@[0.01:0.10]` ist der Mittelwert über den gesamten Durchlauf. Dieser Wert ist zugleich `fitness`, den die Auswahl des besten Checkpoints verwendet. Beide mAP-Schlüssel werden aus den verwendeten Schwellenwerten aufgebaut. Die Übergabe von `dist_thresholds=` benennt sie deshalb um.

`metrics/MLE` ist der mittlere Abstand zugeordneter Paare beim strengsten Schwellenwert in denselben normalisierten Einheiten. `metrics/MAE` und `metrics/RMSE` sind Zähl- statt Lokalisierungsmetriken. Sie messen je Bild die Differenz zwischen der Anzahl vorhergesagter und tatsächlicher Punkte.

FOMO ergänzt darüber eine zweite Gruppe auf Rasterebene. Sie durchläuft Konfidenz und `nms_radius` und veröffentlicht die Kombination mit dem besten F1 als `metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`, `metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP` und `metrics/grid_FN`. Die zugehörigen Einstellungen stehen unter `decode/threshold` und `decode/nms_radius`.

## Export

FOMO verwendet den gemeinsamen Exportpfad. Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

Die Abdeckung je Format steht auf der [FOMO-Seite](/docs/models/fomo) und in der [vollständigen Exportmatrix](/docs/reference/export-matrix). LocateAnything und SenseNova-Vision lassen sich nicht exportieren. `export()` löst für beide einen Fehler aus, da ein generatives Modell keinen aufzeichenbaren Erkennungsgraphen besitzt.
