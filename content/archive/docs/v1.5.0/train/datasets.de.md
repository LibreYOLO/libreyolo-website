---
title: Datensätze
seo_title: Trainingsdatensätze in LibreYOLO
description: >-
  Die von LibreYOLO gelesene Datensatz-YAML, die erwartete Ordnerstruktur,
  automatische Downloads und der Doctor-Befehl zur Prüfung eines Datensatzes vor
  dem Training.
lead: >-
  Ein LibreYOLO-Datensatz ist eine YAML-Datei, die ein Stammverzeichnis, seine
  Aufteilungen und seine Klassennamen angibt. Alles Weitere, einschließlich des
  Speicherorts der Labeldateien, wird nach Konvention aus dieser Datei
  abgeleitet.
keywords:
  - YOLO Datensatzformat
  - data.yaml erstellen
  - eigenen Datensatz trainieren
  - YOLO Labelformat
  - COCO JSON Datensatz
  - Datensatz automatisch herunterladen
  - LibreYOLO Doctor
  - Klassenungleichgewicht prüfen
  - Datenleck Train Val Split
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Ein mitgelieferter Name, ein relativer Pfad oder ein absoluter Pfad
        funktionieren gleichermaßen.

        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Einen Datensatz prüfen
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Einen CI-Job auch bei Warnungen fehlschlagen lassen
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Decodierung der Bilder überspringen
      language: bash
      code: >
        # Liest nur Labels und YAML. Für Prüfungen auf Beschädigungen, Duplikate

        # und Datenlecks zwischen Aufteilungen werden die Pixel benötigt, daher
        entfallen sie.

        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## Datensatz für das Training angeben

`data=` akzeptiert einen YAML-Pfad oder den Namen einer mit dem Paket
ausgelieferten Konfiguration.

<code-tabs name="train" />

Der Name wird in einer festen Reihenfolge aufgelöst: zuerst als vorhandener
absoluter Pfad, dann wie angegeben relativ zum Arbeitsverzeichnis, anschließend
mit angehängtem `.yaml` und zuletzt im Verzeichnis der mitgelieferten
Konfigurationen. Wenn nichts passt, nennt die Fehlermeldung alle durchsuchten
Verzeichnisse und führt die mitgelieferten Konfigurationen auf.

## Mitgelieferte Konfigurationen

Dreizehn Datensatzkonfigurationen werden im Paket unter
`libreyolo/config/datasets/` mitgeliefert.

| Konfiguration | Aufgabe | Hinweise |
|---|---|---|
| `coco8.yaml` | Erkennung | 8 Bilder, Download über eine einfache URL |
| `coco128.yaml` | Erkennung | 128 Bilder |
| `coco1000.yaml` | Erkennung | 800 Training, 200 Validierung |
| `coco5000.yaml` | Erkennung | 4000 Training, 1000 Validierung |
| `coco.yaml` | Erkennung | vollständiger COCO 2017 |
| `coco-val-only.yaml` | Erkennung | nur val2017 |
| `coco8-pose.yaml` | Pose | 8 Bilder, COCO-17-Schlüsselpunkte |
| `coco-pose.yaml` | Pose | COCO-2017-Schlüsselpunkte |
| `ade20k.yaml` | semantisch | 150 Klassen |
| `cityscapes.yaml` | semantisch | 19 Klassen, manueller Download |
| `cocostuff.yaml` | semantisch | 182 Klassen, manueller Download |
| `gopro.yaml` | Wiederherstellung | Paare zur Bewegungsunschärfeentfernung |
| `sr8.yaml` | Wiederherstellung | Paare zur Superauflösung |

Nur `coco8.yaml` und `coco128.yaml` enthalten eine einfache Download-URL. Die
übrigen enthalten entweder einen Python-Downloadblock, für den die unten
beschriebene ausdrückliche Zustimmung erforderlich ist, oder erwarten bereits
auf dem Datenträger vorhandene Daten.

## Speicherort eines Datensatzes

Der YAML-Schlüssel `path` benennt das Stammverzeichnis des Datensatzes. Ein
absoluter `path` wird unverändert verwendet. Nach einem relativen Pfad wird zuerst
im Datensatzverzeichnis und dann neben der YAML-Datei selbst gesucht. Ein
herunterzuladender Datensatz wird im Datensatzverzeichnis abgelegt.

Dieses Verzeichnis ist `~/datasets` und kann durch die Umgebungsvariable
`LIBREYOLO_DATASETS_DIR` überschrieben werden. Dafür gibt es keine
Einstellungsdatei.

## YAML-Schlüssel

```yaml
path: my-dataset        # Datensatzstammverzeichnis
train: images/train     # für das Training erforderlich
val: images/val         # für die Validierung erforderlich
test: images/test       # optional
nc: 3                   # optional; muss mit names übereinstimmen
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # optional
```

`train`, `val` und `test` akzeptieren jeweils ein Bildverzeichnis, eine
`.txt`-Datei mit einem Bildpfad pro Zeile oder eine Liste, die beides mischt.
Zeilen in einer `.txt`-Liste dürfen relativ sein und werden dann relativ zum
Verzeichnis der Listendatei aufgelöst. Mit `#` beginnende Zeilen werden
übersprungen.

`names` kann eine Liste oder eine Zuordnung mit ganzzahligen Schlüsseln sein.
`nc` ist optional. Sind beide vorhanden und stimmen nicht überein, meldet Doctor
dies als Fehler.

## Verzeichnisstruktur und Labeldateien

Erkennung, Segmentierung, Pose und orientierte Boxen verwenden dieselbe Struktur.
Der Labelpfad wird aus dem Bildpfad abgeleitet, indem eine Verzeichniskomponente
`images` in `labels` umgeschrieben und die Erweiterung in `.txt` geändert wird:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Nur eine vollständige Pfadkomponente `images` wird umgeschrieben. Ein Verzeichnis
namens `images_old` bleibt daher unverändert.

Eine Erkennungszeile besteht aus fünf Feldern, die alle anhand der ursprünglichen
Bildbreite und -höhe auf `[0, 1]` normalisiert sind:

```text
<class_id> <cx> <cy> <w> <h>
```

Eine fehlende oder leere Labeldatei bedeutet, dass das Bild keine Objekte
enthält. Es wird als Hintergrund trainiert, statt einen Fehler auszulösen. Eine
Zeile mit mehr als fünf Feldern wird als Polygon gelesen, dessen Box der
Ausdehnung des Polygons entspricht. Ein Segmentierungsexport lässt sich dadurch
ohne Beanstandung für ein Erkennungstraining laden. Doctor meldet, wie viele
Zeilen auf diesem Weg verarbeitet wurden.

## Weitere Aufgaben

Die Segmentierung verwendet dieselbe Struktur mit Polygonzeilen im Format
`<class_id> <x1> <y1> ... <xN> <yN>` und mindestens drei Punkten. Eine
Erkennungszeile mit fünf Feldern ist zulässig und bezeichnet eine rechteckige
Instanz.

Für Pose werden `kpt_shape: [K, D]` und optional eine `flip_idx`-Permutation zur
YAML-Datei hinzugefügt. Jede Zeile enthält genau `5 + K * D` Felder: die Box und
danach `K` Schlüsselpunkte als `x y` oder `x y v`, wobei die Sichtbarkeit `0`,
`1` oder `2` ist.

Orientierte Boxen verwenden genau neun Felder: die Klasse, gefolgt von vier
Eckpunkten in normalisierten Koordinaten. In der Datei wird kein Winkel
gespeichert.

Bei der semantischen Segmentierung wird jedes Bild mit einer einkanaligen Maske
derselben Auflösung gekoppelt. Ihr Pfad entsteht, indem `images` durch
`masks_dir` ersetzt wird, dessen Standardwert `masks` ist. Der Pixelwert `255`
bedeutet „ignorieren“. `label_mapping` ordnet Quell-IDs beim Laden Trainings-IDs
zu.

Die Klassifizierung verwendet statt Labeldateien einen ImageFolder-Baum. `train/`
und `val/` enthalten jeweils ein Verzeichnis pro Klasse. Die Zuordnung von
Klassen zu Indizes folgt der sortierten Reihenfolge der Ordnernamen.

Bei der Wiederherstellung wird über `input_dir` und `target_dir` eine
beeinträchtigte Eingabe mit einem sauberen Ziel gleicher Auflösung gekoppelt.
Tiefe, Oberflächennormalen und Kanten koppeln ein Bild jeweils über einen eigenen
Verzeichnisschlüssel mit einer dichten Karte.

Der vollständige Vertrag für jede Aufgabe, einschließlich der Konventionen für
Tiefenskalierung und der PNG-Codierung panoptischer Segment-IDs, steht in
`docs/dataset_schema.md` im Bibliotheksrepository.

## Natives COCO JSON

Eine COCO-JSON-Anmerkungsdatei kann direkt verwendet werden. Füge eine
`annotations`-Zuordnung hinzu. Der Aufteilungspfad wird dann zum Bildstammverzeichnis:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Wenn `names` vorhanden ist, müssen die JSON-Kategorienamen damit übereinstimmen.
`names` definiert außerdem die vom Modell vorhergesagten Label-IDs. Ohne `names`
werden die COCO-Kategorie-IDs sortiert und dicht auf `0..N-1` abgebildet.

Dieser Pfad erwartet ein Bildverzeichnis pro Aufteilung. Bei einer Pfadliste oder
einer `.txt`-Bildliste tritt ein Fehler auf, statt stillschweigend eine andere
Menge zu laden.

## Automatischer Download

Ein Datensatz gilt als vorhanden, wenn sein `train`- oder `val`-Pfad zu einem
nicht leeren Verzeichnis oder einer vorhandenen Datei aufgelöst wird. Ist das
nicht der Fall und enthält die YAML-Datei einen Schlüssel `download`, bestimmt
dessen Wert den nächsten Schritt.

Eine `http`- oder `https`-URL wird abgerufen und bei einer ZIP-Datei in das
Datensatzstammverzeichnis entpackt. Jeder andere Wert gilt als eingebettetes
Python-Skript und wird nur mit `allow_download_scripts=True` ausgeführt. Ohne
diese Option wird das Skript mit einer Warnung übersprungen und das Training mit
den auf dem Datenträger vorhandenen Daten fortgesetzt.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

Die Option kontrolliert Codeausführung, nicht den Netzwerkzugriff. URL-Downloads
finden in beiden Fällen statt. Nur die Blöcke `download: |` benötigen die
Freigabe. Die CLI gibt bei aktivierter Option eine Warnung aus. Doctor aktiviert
sie nie.

## Datensatz vor dem Training prüfen

`libreyolo doctor` liest einen Erkennungsdatensatz und meldet mögliche Probleme,
bevor eine GPU beteiligt ist. Bei Fehlern endet der Befehl mit Status 1 und kann
daher als CI-Schranke dienen.

<code-tabs name="doctor" />

Die Prüfungen gehören zu sechs Familien:

| Familie | Gesuchte Probleme |
|---|---|
| `config` | fehlendes `names`, von `names` abweichendes `nc`, fehlende oder leere Aufteilungen, doppelte Klassennamen |
| `files` | Bilder ohne Labeldatei, Labels ohne Bild, in einer Aufteilung aufgeführte fehlende Bilder, Kollisionen bei Dateistämmen |
| `labels` | fehlerhafte Zeilen, Klassen-IDs außerhalb von `[0, nc)`, Koordinaten außerhalb von `[0, 1]`, Boxen ohne Fläche, winzige oder riesige Boxen, doppelte Boxen, bytegenau identische Labeldateien |
| `balance` | Klassen ohne oder mit wenigen Instanzen, Verhältnis des Klassenungleichgewichts, nur in einer Aufteilung vertretene Klassen, Anteil der Hintergrundbilder |
| `images` | nicht decodierbare Dateien, EXIF-Drehung, ungewöhnliche Kanalanordnungen, einheitliche Bilder, exakte und nahezu identische Duplikate |
| `splits` | dasselbe Bild in zwei Aufteilungen, exakt oder nahezu identisch |

`--only` und `--skip` akzeptieren eine Prüf-ID oder ein Familienpräfix. Damit ist
`skip=images,labels.tiny_object` gültig. `--fast` entfernt alle Prüfungen, für die
Pixel decodiert werden müssen, also die Familien `images` und `splits`.

Zwei Verhaltensweisen sind wichtig. Mit `--strict` führen neben Fehlern auch
Warnungen zu einem fehlgeschlagenen Exit-Code. Außerdem deckt Doctor nur
Erkennungsdatensätze ab. Ein Pose-, Segmentierungs- oder Datensatz mit orientierten
Boxen wird mit einer Meldung zum erkannten Typ abgelehnt, statt anhand des
falschen Vertrags geprüft zu werden.

## Verwandte Themen

- [Hyperparameter](/docs/train/hyperparameters) beschreibt die Argumente, die
  `train()` nach dem Einrichten der Daten akzeptiert.
- [Validierung und Metriken](/docs/train/validation) erklärt die Auswertung der
  Aufteilung `val` oder `test`.
