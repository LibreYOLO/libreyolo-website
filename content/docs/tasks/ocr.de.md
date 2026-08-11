---
title: OCR
seo_title: 'OCR: Texterkennung und Textlesen in LibreYOLO'
description: >-
  Finde und lies mit LibreYOLO Text in Bildern. Sage Vierecke und Transkripte
  voraus, beschrifte einen JSONL-Datensatz und validiere mit Hmean,
  End-to-End-F1 und 1-NED.
lead: >-
  OCR lokalisiert Text in einem Bild und liest ihn. LibreYOLO stellt dies als
  ocr-Aufgabe bereit. Sie gibt pro Textregion ein Polygon aus vier Punkten und
  ein Transkript in Lesereihenfolge zurück.
keywords:
  - ocr python bibliothek
  - szenentext erkennen
  - text-erkennung vierecke
  - PP-OCRv5 python
  - end-to-end text spotting
last_verified: 1.5.0
snippets:
  predict:
    - label: Text in einem Bild lesen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die t-Stufe ist die leichtere der beiden und für CPUs ausgelegt.
        # SAMPLE_IMAGE hält das Beispiel ausführbar; nutze ein eigenes Textbild.
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Vierecke auslesen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(regions.data.shape)   # (N, 4, 2) Polygone, OL OR UR UL
        print(regions.xyxy)         # achsenparallele Hüllen dieser Polygone
        print(regions.det_conf)     # Erkennungs-Score, getrennt von .conf
    - label: Nach Recognition-Confidence filtern
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # Mit Positionen statt boolescher Maske indizieren: Beim Slicing werden
        # Transkripte und beide Score-Arrays mit der Geometrie übernommen.
        regions = result.ocr.numpy()
        keep = regions[np.flatnonzero(regions.conf >= 0.9)]
        print(keep.texts)
  val:
    - label: Validieren und Metrikschlüssel auslesen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # Fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## Definition

Die Aufgabe `ocr` führt in einem Aufruf zwei Schritte aus: Sie lokalisiert jede
Textregion in einem Bild und transkribiert sie. Die Regionen werden als
Polygone aus vier Punkten und nicht als achsenparallele Boxen zurückgegeben,
weil Text in Szenen häufig gedreht ist. Ihre Reihenfolge entspricht der
Leserichtung, zunächst von oben nach unten und anschließend von links nach
rechts.

Eine Vorhersage füllt `result.ocr`, eine `OCRRegions`-Payload. `.data` ist ein
Float-Array der Form `(N, 4, 2)` mit Polygonen in Pixelkoordinaten des
Originalbilds. Die Punkte sind oben links, oben rechts, unten rechts und unten
links angeordnet. `.texts` ist die Liste der N Transkripte. `.conf` enthält den
Recognition-Score pro Region und `.det_conf` den Erkennungs-Score. `.xyxy`
liefert die achsenparallele Hülle jedes Polygons. Da die Vierecke echte
Polygone sind, füllen sie `result.boxes` nicht. Beim Slicing von `OCRRegions`
werden die Transkripte und beide Score-Arrays zusammen mit der Geometrie
übernommen.

## Modelle

Zwei Familien unterstützen `ocr`.

[PP-OCRv5](/docs/models/pp-ocrv5) ist die dafür vorgesehene Pipeline: Ein
Detektor mit differenzierbarer Binarisierung findet die Textvierecke, die
anschließend von einem SVTR/CTC-Recognizer gelesen werden. Beide Stufen und der
Zeichensatz für die Erkennung sind in einer `.pt`-Datei gebündelt. Es gibt eine
leichtere Stufe für CPUs und eine Server-Stufe mit höherer Accuracy. Ein
Dictionary deckt vereinfachtes und traditionelles Chinesisch, Englisch,
Japanisch und Pinyin ab.

[SenseNova-Vision](/docs/models/sensenova-vision) führt OCR aus, indem es mit
demselben 7B-Checkpoint, der sechs weitere Aufgaben unterstützt, Wörter als
getaggten Text generiert. Geladen wird es mit
`LibreVLM("sensenova-vision", task="ocr")`. Es benötigt das Extra `sensenova`,
und seine Gewichte sind auf die nichtkommerzielle Nutzung beschränkt. Die
Lizenz wird auf seiner Seite beschrieben.

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert.

<code-tabs name="predict" />

PP-OCRv5 führt die Erkennung mit einer festen Begrenzung der langen Bildseite
aus und liest die zugeschnittenen Regionen anschließend in Batches.
`rec_batch` steuert, wie viele Ausschnitte pro Forward Pass durch den Recognizer
laufen. Quellen mit mehreren Bildern werden sequenziell verarbeitet, da eine
zweistufige Pipeline Bilder nicht stapelübergreifend in Batches verarbeitet.
Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen,
Streaming und Ergebnisverarbeitung.

## Datensatzformat

OCR-Labels werden für jeden Split in einer JSONL-Datei gespeichert. Darin
steht ein JSON-Objekt pro Bild. Die Dateien befinden sich neben den Bildern.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Jede Zeile benennt ein Bild und führt seine Regionen auf:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` ist ein Viereck mit absoluten Pixelkoordinaten. Seine Punkte sind
oben links, oben rechts, unten rechts und unten links angeordnet. Eine Region
mit nicht lesbarem Text erhält das Label `"text": "###"`. Dies ist die
ICDAR-Konvention für nicht zu berücksichtigende Bereiche: Die Region wird von
der Recognition-Bewertung ausgeschlossen. Eine sie überlappende Vorhersage
wird ignoriert und nicht als False Positive gezählt.

Es genügt, das Stammverzeichnis als `data=` zu übergeben. Alternativ kannst du
eine Datensatz-YAML verwenden. Sie enthält `path` sowie optionale Namen für die
Verzeichnisse `images` und `labels`. `nc: 1` und `names: {0: text}` dienen als
Schema-Platzhalter, da ein OCR-Modell `Results.ocr` statt Erkennungen
zurückgibt. Unter [Datensatzformate](/docs/reference/dataset-formats) findest du
den vollständigen Vertrag.

## Training

Keine der beiden OCR-Familien implementiert Training. `train()` löst bei
beiden `NotImplementedError` aus. Die OCR-Unterstützung umfasst nur Vorhersage
und Validierung. Die Seite zu PP-OCRv5 nennt den Apache-2.0-lizenzierten
Upstream-Trainingscode und das Konvertierungsskript, mit dem du einen
nachtrainierten Checkpoint wieder in LibreYOLO importierst.

## Validierung

`val()` bewertet die gesamte Pipeline aus Erkennung und Texterkennung. Dabei
werden vorhergesagte Polygone eindeutig Ground-Truth-Polygonen mit einer IoU
über 0.5 zugeordnet.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall` und `metrics/det_hmean` bewerten
nur die Lokalisierung. Für eine Übereinstimmung genügt die
Polygonüberlappung, unabhängig vom Transkript. `metrics/e2e_precision`,
`metrics/e2e_recall` und `metrics/e2e_f1` beziehen das Lesen ein. Hier sind
sowohl dieselbe Polygonüberlappung als auch ein exakt übereinstimmendes
Transkript nach NFKC-Normalisierung und Entfernung von Leerraum erforderlich.
Beim Vergleich wird die Groß- und Kleinschreibung berücksichtigt.
`metrics/e2e_f1` ist zugleich `fitness`, also der Wert für die Auswahl des
besten Checkpoints.

`metrics/rec_1-NED` bewertet den Recognizer separat anhand der bereits durch
die Erkennung zugeordneten Paare. Der Wert entspricht eins minus der
normalisierten Editierdistanz. Ein Transkript mit einem falschen Zeichen liegt
daher nahe bei 1, während der End-to-End-F1 dafür 0 ergibt.

## Export

Für diese Aufgabe ist kein Exportformat verfügbar. PP-OCRv5 besteht aus zwei
gemeinsam arbeitenden Netzen und nicht aus einem einzelnen nachverfolgbaren
Graphen. `export()` löst bei beiden Familien für jedes Format einen Fehler aus.
Wenn du das Modell außerhalb von LibreYOLO ausrollen möchtest, führe das
Fine-Tuning im Upstream-Projekt durch und verwende dessen Deployment-Pfad.

