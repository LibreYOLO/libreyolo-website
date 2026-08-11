---
title: Grundkonzepte
seo_title: Grundkonzepte von LibreYOLO
description: >-
  So hängen Aufgaben, Modellfamilien, Größen und Checkpoint-Dateinamen in
  LibreYOLO zusammen, und diese Zusagen macht jede Support-Stufe.
lead: >-
  Vier Konzepte beschreiben jedes Modell in LibreYOLO: seine Aufgabe, seine
  Familie, seine Größe innerhalb dieser Familie und die Support-Stufe der
  Familie. Der Checkpoint-Dateiname codiert die ersten drei.
keywords:
  - libreyolo grundkonzepte
  - libreyolo aufgaben
  - libreyolo modellfamilien
  - libreyolo checkpoint namen
  - libreyolo support-stufen
last_verified: 1.5.0
meta:
  - label: Dateinamenschema
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: Kanonische Aufgaben
    value: 17
  - label: Support-Stufen
    value: 'Flagship, Core, Supported, Inference only, Museum, Sibling tier'
snippets:
  inspect:
    - label: Familien auflisten
      language: bash
      code: |
        # Aufgaben, Größen und Eingabeauflösungen aller registrierten Familien.
        libreyolo models
    - label: Einzelnes Modell
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Aufgabe auswählen
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Aliasse werden an der API-Grenze normalisiert: "keypoints" wird zu
        # "pose", "det" zu "detect", "semantic-segmentation" zu "semantic".
        model = LibreYOLO("LibreYOLO9t.pt", task="det")
        print(model.task)
source_hash: 23d045463a6a8411
---

## Aufgaben

Eine Aufgabe bestimmt, was ein Modell zurückgibt. LibreYOLO besitzt 17
kanonische Aufgabennamen. Jeder bezeichnet das Feld im `Results`-Objekt, das
die jeweilige Ausgabe enthält.

| Aufgabe | Rückgabe |
|---|---|
| `detect` | Achsenparallele Boxen mit Klasse und Confidence |
| `segment` | Masken pro Instanz, eine Maske je erkanntem Objekt |
| `semantic` | Ein Klassenlabel pro Pixel ohne Trennung von Instanzen |
| `panoptic` | Ein nicht überlappendes Label pro Pixel, das zählbare Things mit formlosem Stuff vereint |
| `pose` | Keypoints pro Instanz, deren Zeilen an den Boxen ausgerichtet sind |
| `classify` | Eine Wahrscheinlichkeitsverteilung über einen Labelsatz für das gesamte Bild |
| `obb` | Orientierte Boxen mit Rotationswinkel |
| `point` | Eine Bildkoordinate pro Erkennung statt einer Box |
| `depth` | Eine dichte Karte relativer inverser Tiefe |
| `normal` | Ein dichtes Feld von Einheitsvektoren für Oberflächennormalen |
| `edge` | Eine dichte Kantenwahrscheinlichkeitskarte |
| `restore` | Ein restauriertes RGB-Bild für Entschärfung, Entrauschen oder Super-Resolution |
| `matte` | Eine weiche Vordergrundkarte von 0 bis 1 für die Hintergrundentfernung |
| `ocr` | Textvierecke mit Transkripten in Lesereihenfolge |
| `embed` | Ein L2-normalisierter Vektor, dessen Skalarprodukt die Übereinstimmung misst |
| `gaze` | Eine Blickrichtung pro erkanntem Gesicht |
| `mesh` | Ein positionierter 3D-Körper pro erkannter Person |

Diese Namen stehen in Checkpoint-Metadaten und Dateinamen. Vertraute Aliasse
werden überall akzeptiert, wo du eine Aufgabe übergibst, und vor jedem weiteren
Schritt normalisiert. `detection` und `det` werden zu `detect`, `keypoints` zu
`pose`, `cls` zu `classify`, `deblur`, `denoise` und `super-resolution` zu
`restore`, `face-recognition` und `reid` zu `embed`. Ein unbekannter Name löst
einen Fehler aus, statt unbemerkt den Standard zu wählen.

`segment`, `semantic` und `panoptic` sind drei unterschiedliche Aufgaben und
keine drei Bezeichnungen für dasselbe Konzept. Instanzmasken, Pixelklassen und
die kombinierte Thing-und-Stuff-Karte verwenden unterschiedliche Ground Truth,
Metriken und Ergebnisfelder.

## Modellfamilien

Eine Familie ist eine Architekturlinie mit eigenem Code für Laden,
Vorverarbeitung und Nachverarbeitung. Jede Familie deklariert einen
`FAMILY`-Bezeichner wie `yolo9`, `rfdetr` oder `dfine`, ihre unterstützten
Aufgaben und die Eingabeauflösung jeder angebotenen Größe.

`LibreYOLO()` ist eine Factory und keine Klasse. Sie lädt einen übergebenen
Pfad, identifiziert die Familie anhand der Checkpoint-Metadaten oder ersatzweise
der Tensorschlüssel und gibt eine Instanz des Modells dieser Familie zurück.
Deshalb erfordert der Wechsel des Detektors nur eine geänderte Zeile. Das
zurückgegebene Objekt stellt dieselben Methoden `predict`, `train`, `val` und
`export` bereit und gibt denselben Typ `Results` zurück.

<code-tabs name="inspect" />

Eine Familie mit mehreren Aufgaben veröffentlicht normalerweise einen
getrennten Checkpoint pro Aufgabe, häufig mit einer anderen Größenauswahl. In
einigen Fällen teilen sich zwei Runtime-Aufgaben ein Artefakt. In beiden Fällen
bilden die unterstützten Aufgaben eine feste Liste. Eine Anfrage außerhalb
dieser Liste löst einen Fehler aus, der die unterstützten Aufgaben nennt,
anstatt etwas nur annähernd Passendes zu laden.

Die vollständige Liste mit Benchmarks und veröffentlichten Gewichten pro
Familie findest du unter [Alle Modelle](/docs/models).

## Größen

Eine Größe ist eine Variante innerhalb einer Familie. Sie wird als
kleingeschriebener Code direkt an das Familienpräfix angehängt. Häufige
Buchstaben sind `n` für Nano, `t` für Tiny, `s` für Small, `m` für Medium, `l`
für Large und `x` für XLarge. Die Codes sind jedoch familienspezifisch. Mehrere
Familien verwenden völlig andere Formen: nach dem Backbone benannte Codes wie
`r50` oder `r101`, bei denen die Größe eine ResNet-Tiefe angibt,
Compound-Scaling-Codes von `b0` bis `b3` oder einen Namen für den einzigen
veröffentlichten Checkpoint. YOLOv9 verwendet `c` für Compact, während andere
Familien an dieser Stelle `l` verwenden.

Die Größe legt auch die Eingabeauflösung fest. Bei Familien mit mehreren
Aufgaben kann sich die Auflösung je nach Aufgabe unterscheiden. Beides wird aus
der Familie gelesen und nie angenommen. `libreyolo models` gibt die Werte aus.

## Checkpoint-Dateinamen

Jede veröffentlichte Gewichtsdatei folgt diesem Schema:

```text
Libre<FAMILY><size>[-<task>].pt
```

Das Familienpräfix ist pro Familie fest. Die kleingeschriebene Größe wird ohne
Trennzeichen angehängt, das Aufgabensuffix mit einem Bindestrich. Die Erkennung
besitzt gemäß der üblichen YOLO-Checkpoint-Konvention kein Suffix.
`LibreYOLO9t.pt` ist daher ein Detektor und `LibreRFDETRn-seg.pt` ein
Segmentierungsmodell derselben Familie.

| Aufgabe | Suffix |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

Eine Familie ohne suffixlose Aufgabe kann das Suffix verlangen. Ein Name ohne
Suffix ist dann kein gültiger Checkpoint. Wenn eine Familie Gewichte
veröffentlicht, die auf einem anderen als ihrem Standarddatensatz trainiert
wurden, hängt sie den Datensatznamen als weiteres Suffix an. Diese Variante
bleibt Teil des Repository-Namens, aus dem die Datei heruntergeladen wird.

Drei Stufen liegen außerhalb dieses Schemas. Familien für promptbasierte
Segmentierung, Vision-Language-Familien und Open-Vocabulary-Detektoren werden
nicht in der Checkpoint-Factory registriert und erzeugen keine Datei
`Libre<FAMILY><size>.pt`. Ihr Präfix bezeichnet stattdessen einen
heruntergeladenen Snapshot von Hugging Face oder einen promptbasierten
Checkpoint. Die Groß- und Kleinschreibung der Upstream-Marke bleibt dort
absichtlich erhalten.

## Ermittlung der Aufgabe

Wenn mehrere Signale eine Aufgabe benennen könnten, werden sie in einer festen
Reihenfolge geprüft. Der erste vorhandene Wert gilt: zunächst das übergebene
Argument `task`, dann die in den Checkpoint-Metadaten gespeicherte Aufgabe,
anschließend das Aufgabensuffix im Dateinamen und zuletzt die Standardaufgabe
der Familie. Das Ergebnis wird vor dem Erstellen des Modells anhand der
unterstützten Aufgaben der Familie geprüft. Eine Abweichung scheitert daher
beim Laden und erzeugt nicht erst später eine falsche Ausgabe.

## Support-Stufen

Jede Familie gehört genau einer Stufe an. Eine Stufe beschreibt den Umfang der
technischen Betreuung und nicht die Accuracy. Sie gibt an, wo eine neue
Funktion zuerst implementiert wird und welche Pfade dauerhaft grün gehalten
werden.

| Stufe | Bedeutung |
|---|---|
| Flagship | Funktionen werden zuerst hier entwickelt und vollständig auf GPUs validiert |
| Core | Zentrale trainierbare Detektoren. Funktionen folgen den Flagships in derselben Release-Welle |
| Supported | Ergänzende trainierbare Familien. Werden im CI-System grün gehalten, Funktionen werden nach Gelegenheit ergänzt |
| Inference only | Vorhersage, Validierung und Export. Trainingsfunktionen gelten nicht |
| Museum | Eine eingefrorene Ausstellung. Nur Fehlerbehebungen |
| Sibling tier | Eine getrennte Produktoberfläche mit eigener Factory und eigenem Vertrag |

Jede Modellseite zeigt die Stufe ihrer Familie im Kopfbereich. Die beiden
Flagship-Familien sind [YOLOv9](/docs/models/yolov9) für CNN-Detektoren und
[RF-DETR](/docs/models/rf-detr) für Transformer-Detektoren. Beginne dort,
sofern kein besonderer Grund für eine andere Wahl besteht.

„Inference only“ bezeichnet die fehlende Funktion: eine Trainingsschleife in
LibreYOLO. Vorhersage, Validierung und, sofern von der Familie unterstützt,
Export funktionieren. `train()` löst bei einer solchen Familie einen
`NotImplementedError` aus, der den Grund nennt.

