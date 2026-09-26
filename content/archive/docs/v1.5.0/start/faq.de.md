---
title: FAQ
seo_title: LibreYOLO-FAQ
description: >-
  Kurze Antworten auf Fragen, die alle LibreYOLO-Modelle betreffen: Hardware,
  Lizenzierung, Gewichte, Geräte, Training, Exportabdeckung und CLI.
lead: >-
  Antworten auf Fragen, die nicht nur eine Modellfamilie betreffen.
  Familienspezifische Informationen stehen auf der Seite der jeweiligen Familie.
keywords:
  - libreyolo faq
  - libreyolo gpu erforderlich
  - libreyolo lizenz
  - libreyolo gewichte speicherort
  - libreyolo cli
  - libreyolo offline
last_verified: 1.5.0
source_hash: a729b43a6642f2a0
---

## Mit welchem Modell sollte ich beginnen?

YOLOv9 als CNN-Detektor und RF-DETR als Transformer-Detektor. Beide gehören zur
Flagship-Stufe. Funktionen werden zuerst für sie entwickelt und auf GPUs
validiert. Weitere Informationen findest du unter
[YOLOv9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr) oder in der
Übersicht [aller Modelle](/docs/models).

## Benötige ich eine GPU?

Nein. Jedes Modell läuft auf einer CPU, und alle Beispiele im
[Schnellstart](/docs/quickstart) sind dafür ausgelegt. Eine GPU beeinflusst die
Dauer von Training und Video-Inferenz, nicht deren grundsätzliche
Funktionsfähigkeit.

## Wie wählt LibreYOLO ein Gerät aus?

Der Standardwert ist `device="auto"`. Damit wird CUDA verwendet, wenn PyTorch
es als verfügbar meldet, anschließend Metal Performance Shaders, sofern diese
verfügbar sind, und andernfalls die CPU. Um ein Gerät festzulegen, übergib
`device` an das Modell oder an `predict`, `train`, `val` und `export`.
Akzeptiert werden `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, eine einfache
Ganzzahl wie `0` oder ein Ziffernstring. Die beiden letzten Formen werden zu
`cuda:<n>` erweitert.

`libreyolo checks` gibt den Torch-Build, seine CUDA- und cuDNN-Versionen sowie
jede sichtbare GPU aus. Wenn dieser Befehl kein CUDA anzeigt, ist das
PyTorch-Wheel ein CPU-Build. Unter [Installation](/docs/install) erfährst du,
wie du es ersetzt.

## Wo werden heruntergeladene Gewichte gespeichert?

Im Verzeichnis `weights/` relativ zum Arbeitsverzeichnis. Eine Modellreferenz
ohne Verzeichniskomponente wird dort aufgelöst und bei der ersten Verwendung
heruntergeladen. Eine Referenz mit Verzeichnis wird genau wie angegeben
verwendet und nie abgerufen. Weitere Informationen findest du unter
[Checkpoints und Gewichte](/docs/weights).

## Kann LibreYOLO ohne Netzwerkzugriff ausgeführt werden?

Ja. Rufe die Checkpoints einmal auf einem verbundenen Rechner ab und übertrage
das Verzeichnis `weights/`. Danach greift LibreYOLO nicht mehr auf das Netzwerk
zu. Ein gemeinsam genutzter schreibgeschützter Pfad funktioniert ebenfalls, da
eine Referenz mit Verzeichnis wörtlich übernommen wird. Datensätze werden unter
`~/datasets` oder im Pfad aus `LIBREYOLO_DATASETS_DIR` aufgelöst.

## Kann ich LibreYOLO kommerziell nutzen?

Der Code steht unter der MIT-Lizenz. Für vortrainierte Gewichte gilt dies nicht
automatisch. Sie können Bedingungen des Projekts oder Datensatzes übernehmen,
aus dem sie stammen. Diese Bedingungen sind selbst innerhalb einer Familie
nicht einheitlich. Maßgeblich ist die Lizenz des konkreten Repositorys auf
Hugging Face. Jede Modellseite enthält einen Lizenzierungsabschnitt, der sie
wiedergibt. Bei eingeschränkten Gewichten zeigt LibreYOLO die Einschränkung vor
Beginn des Downloads an.

## Kann ich einen Checkpoint aus einem anderen Projekt laden?

In der Regel ja. Übergib seinen Pfad an `LibreYOLO()`. Erkannte
Upstream-Layouts werden beim Laden konvertiert. Klassenzahl und Namen bleiben
erhalten, und neben der Quelldatei wird ein LibreYOLO-Checkpoint geschrieben.
Unter [Vorhandene Gewichte importieren](/docs/migrate) erfährst du, welche
Formate erkannt werden und welche ein Konvertierungsskript benötigen.

## Warum löst train einen NotImplementedError aus?

Weil diese Familie nur Inferenz unterstützt. Die Ausnahme nennt den Grund.
Vorhersage, Validierung und, sofern unterstützt, Export funktionieren. Für
diese Architektur gibt es in LibreYOLO jedoch keine Trainingsschleife. Die
Support-Stufe im Kopfbereich einer Modellseite informiert dich darüber, bevor
du es versuchst. Siehe [Grundkonzepte](/docs/concepts).

## Was gibt val zurück?

Ein einfaches Dictionary und kein Objekt. Zu den Erkennungsschlüsseln gehören
`metrics/precision`, `metrics/recall`, `metrics/mAP50` und
`metrics/mAP50-95`. Andere Aufgaben geben die für sie relevanten Schlüssel
zurück, etwa `metrics/accuracy_top1` für Klassifikation oder `metrics/PQ`,
`metrics/SQ` und `metrics/RQ` für panoptische Segmentierung.

## Wie verarbeite ich einen Ordner, ein Video oder eine Webcam?

Übergib die Quelle. Ein Dateipfad steht für ein Bild, ein Verzeichnis für alle
darin enthaltenen Bilder, ein Videopfad für ein Video und eine Ganzzahl für
einen Webcam-Index. Eine RTSP-, RTMP-, TCP-, UDP- oder HLS-URL bezeichnet einen
Livestream. Eine `.streams`-Datei führt mehrere Quellen auf. Live-Quellen
erfordern `stream=True`. Dadurch wird pro Frame ein `Results`-Objekt geliefert,
statt eine Liste aufzubauen. Dieses Flag lohnt sich auch für lange Videos und
große Verzeichnisse. Nur YouTube-Seiten-URLs benötigen mit
`libreyolo[stream]` ein Extra.

## Wie behalte ich nur bestimmte Klassen?

Übergib die gewünschten Klassenindizes mit `classes` an `predict`, zum Beispiel
`classes=[0, 2]`. `conf` legt den Confidence-Schwellenwert fest, standardmäßig
`0.25`. `max_det` begrenzt die Erkennungen pro Bild, standardmäßig auf `300`.

## Verwendet die CLI Flags oder key=value-Paare?

Für jeden Befehl werden Schlüssel und Wert durch ein Gleichheitszeichen verbunden:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` akzeptiert einen Pfad oder einen Kurznamen der Form `family-size`,
optional mit Aufgabensuffix. `libreyolo models` führt alle gültigen Namen auf.
Diagnose- und Inventarbefehle akzeptieren außerdem `--json`. Damit werden
dieselben Daten als maschinenlesbares Objekt auf stdout ausgegeben.

## Kann jedes Modell in jedes Format exportiert werden?

Nein. Die Unterstützung gilt pro Familie und Aufgabe und ist nicht einheitlich.
Außerdem muss für jedes Format ein eigenes Extra installiert werden. Jede
Modellseite enthält die Exportmatrix ihrer Familie. Der
[Exportabschnitt](/docs/export) beschreibt die Formate selbst.

## Worin unterscheiden sich segment, semantic und panoptic?

Es sind drei getrennte Aufgaben. `segment` erzeugt eine Maske pro erkanntem
Objekt. `semantic` weist jedem Pixel eine Klasse zu, trennt aber keine
Instanzen. `panoptic` gibt jedem Pixel genau ein Label und vereint zählbare
Things mit formlosem Stuff. Sie verwenden unterschiedliche Ground Truth,
Ergebnisfelder und Metriken. Eine Familie unterstützt die Aufgaben, die in
ihrer Aufgabenliste aufgeführt sind.

## Wie trainiere ich mit eigenen Klassen?

Erstelle eine Datensatz-YAML mit `train`, `val` und `names`. Die Labels liegen
parallel zu den Bildern in einem `labels/`-Verzeichnisbaum. Pro Bild gibt es
eine `.txt`-Datei mit normalisierten Koordinaten. `nc` ist optional und muss,
falls vorhanden, mit `names` übereinstimmen. Führe zuerst
`libreyolo doctor <data.yaml>` aus. Der Befehl prüft den Datensatz auf Probleme
und beendet sich bei Fehlern mit einem von null verschiedenen Status. Dadurch
kannst du ihn als CI-Gate verwenden.

## Warum erscheint beim Laden eine Metadatenwarnung?

Weil der Checkpoint keine vollständigen v1.0-Metadaten enthält. Das Laden wird
über einen Kompatibilitätspfad fortgesetzt. Die Warnung nennt genau die
fehlenden Schlüssel. Führe `libreyolo metadata path=<file>` aus, um die
vorhandenen Metadaten anzuzeigen. Unter
[Checkpoints und Gewichte](/docs/weights) erfährst du, welche Angaben das Schema
verlangt.

## Ein Import funktioniert nach einem Upgrade nicht mehr. Was hat sich geändert?

Zwei Klassennamen wurden aus Konsistenzgründen umbenannt: `LibreYOLORTDETR`
wurde zu `LibreRTDETR` und `LibreYOLORFDETR` zu `LibreRFDETR`. Die alten Namen
werden weiterhin aufgelöst und geben einen `DeprecationWarning` mit einem
Verweis auf den neuen Namen aus. Bestehender Code läuft daher weiter, während
du ihn aktualisierst.

