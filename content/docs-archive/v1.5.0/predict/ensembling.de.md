---
title: Detektor-Ensembles
seo_title: Detektor-Ensembles in LibreYOLO
description: >-
  Führe mehrere Detektoren für ein Bild aus und fusioniere ihre Boxen mit
  Weighted Boxes Fusion oder NMS, auch bei Modellen mit unterschiedlichen
  Klassenlisten.
lead: >-
  LibreEnsemble führt zwei oder mehr Detektoren für dasselbe decodierte Bild aus
  und fusioniert ihre Boxen zu einem Results-Objekt. Die Mitglieder behalten
  ihre eigenen Gewichte, Schwellenwerte, Geräte und Klassenlisten.
keywords:
  - Modell-Ensemble Objekterkennung
  - Weighted Boxes Fusion
  - WBF Python
  - zwei Detektoren kombinieren
  - Begrenzungsrahmen fusionieren
  - LibreEnsemble
  - Ensemble Objekterkennung Python
  - min_votes
last_verified: 1.5.0
verification: >-
  Konstruktor- und Aufrufsignaturen, Standardwerte, Validierungsfehler,
  Vereinheitlichung des Klassenraums, Stimmenzählung und zurückgegebenes
  Results-Objekt aus libreyolo/ensemble/model.py. Fusionsalgorithmen und ihre
  Argumente aus libreyolo/ops/fusion.py. Entwurfsabsicht aus
  docs/adr/0004-model-ensembling.md. Verwendungsmuster mit
  tests/unit/test_ensemble.py und tests/unit/test_ops_fusion.py abgeglichen.
snippets:
  basic:
    - label: Zwei fusionierte Detektoren
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # Mitglieder können Checkpoint-Pfade oder bereits geladene Modelle sein.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Gewichte und erforderliche Stimmen
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # üblicherweise proportional zur Validierungs-mAP
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # nur Boxen behalten, die beide Mitglieder gefunden haben
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Schwellenwerte je Mitglied
      language: python
      code: >
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE


        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])


        # Ein Skalar gilt für alle Mitglieder; eine Liste wird je Mitglied
        gelesen.

        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)

        print(len(result.boxes))
  external:
    - label: Einen nicht von LibreYOLO geladenen Detektor einbinden
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # (boxes, scores, labels) zurückgeben: xyxy in Pixeln des Originalbildes.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: Dieselben Quellen wie bei einem einzelnen Modell
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Ersetze clip.mp4 durch eine lokale Videodatei.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 6dcd2f84ec6f3f65
---

## Funktionsweise eines Ensembles

`LibreEnsemble` übernimmt zwei oder mehr Detektoren, führt jeden für dasselbe Bild aus und fusioniert ihre Boxen zu einem einzigen `Results`-Objekt. Es ist ein Konstrukt für den Vorhersagezeitpunkt. Es gibt nichts zu trainieren und die Mitglieder bleiben unabhängige Modelle, die sich einzeln validieren und exportieren lassen.

Unterstützt wird ausschließlich die Objekterkennung. Ein Mitglied mit einer anderen Aufgabe löst bei der Erstellung `ValueError` aus. Die Fehlermeldung nennt den Index und die Aufgabe des Mitglieds.

Beide Namen werden verzögert importiert und verursachen daher erst bei ihrer Verwendung Kosten:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Erstellung

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` ist eine Sequenz aus mindestens zwei Elementen. Ein Eintrag vom Typ `str` oder `Path` wird über `LibreYOLO()` geladen. Alle anderen Einträge müssen aufrufbar sein und ein Dictionary `names` bereitstellen. Weniger als zwei Mitglieder lösen `ValueError` aus. Die Übergabe eines einzelnen Strings löst `TypeError` aus, statt über dessen Zeichen zu iterieren.

`weights` ist standardmäßig `None`, was einer gleichmäßigen Gewichtung entspricht. Angegebene Gewichte müssen für jedes Mitglied vorhanden und strikt positiv sein. Ein Gewicht von null löst einen Fehler aus, statt ein Mitglied unbemerkt zu entfernen. Üblicherweise werden die Gewichte proportional zur Validierungs-mAP jedes Mitglieds festgelegt.

`fusion_iou` ist standardmäßig `0.55`. Dieser IoU-Wert bestimmt, ab wann Boxen verschiedener Mitglieder zu einem Cluster zusammengefasst werden. Er unterscheidet sich vom aufrufspezifischen Wert `iou`, der die eigene NMS-Einstellung jedes Mitglieds steuert.

`min_votes` ist standardmäßig `1`, sodass ein einzelnes Mitglied eine Box beisteuern kann. Ein höherer Wert behält nur Cluster bei, die von entsprechend vielen unterschiedlichen Mitgliedern bestätigt wurden. Der Wert muss eine positive Ganzzahl sein und darf die Mitgliederanzahl nicht überschreiten. Für jede Klasse ist er auf die Anzahl der Mitglieder begrenzt, die diese Klasse tatsächlich kennen. Dadurch wird eine Klasse, auf die nur ein Mitglied trainiert wurde, nicht unbemerkt entfernt.

## Fusionsmethoden

Drei Namen sowie eine eigene aufrufbare Funktion werden akzeptiert.

| `fusion` | Verhalten |
|---|---|
| `"wbf"` | Sequenzielle Weighted Boxes Fusion entsprechend dem Paper [1]. Der Standard |
| `"wbf_seeded"` | Weighted Boxes Fusion in einem Durchlauf; klassenbewusstes NMS wählt die Cluster-Startpunkte |
| `"nms"` | Boxen aller Mitglieder zusammenfügen und anschließend klassenbewusstes NMS ausführen |

[1] Roman Solovyev, Weimin Wang, Tatiana Gabruseva, ["Weighted boxes fusion:
Ensembling boxes from different object detection models"](https://arxiv.org/abs/1910.13302),
arXiv:1910.13302.

Weighted Boxes Fusion mittelt die Koordinaten eines Clusters, gewichtet nach Konfidenz. Dadurch entsteht eine Box, die kein einzelnes Mitglied vorgeschlagen hat. Die beiden gewichteten Varianten stimmen bei eindeutigen Clustern überein und können sich bei Ketten überlappender Cluster leicht unterscheiden. `"nms"` wählt statt einer Mittelung einen verbleibenden Kandidaten aus. Dieser behält seine ursprüngliche Bewertung und die Gewichte beeinflussen nur, welche Box gewinnt. Da diese Methode auswählt statt Cluster zu bilden, kann sie keine Stimmen zählen. Die Kombination aus `fusion="nms"` und einem `min_votes` größer als `1` löst `ValueError` aus.

Weighted Boxes Fusion skaliert die Bewertung eines Clusters anhand des Anteils des Mitgliedergewichts, der ihn unterstützt. Bei zwei gleich gewichteten Mitgliedern behält eine nur von einem Modell gefundene Box die Hälfte ihrer Bewertung. Aus `0.9` wird `0.45`. Eine fusionierte Konfidenz kann dadurch unter dem Wert `conf` liegen, mit dem jedes Mitglied ausgeführt wurde. Filtere anhand der fusionierten Bewertung und gehe nicht davon aus, dass der Schwellenwert der Mitglieder weiterhin gilt.

## Mitglieder mit unterschiedlichen Klassenlisten

Die Mitglieder müssen keine gemeinsame Klassenliste besitzen. Ihre Labelräume werden nach Namen vereinigt und jedes Mitglied erhält eine Nachschlagetabelle, die seine Klassen-IDs auf diese Vereinigung abbildet. `ensemble.names` enthält den vereinigten Klassenraum, der auch im zurückgegebenen `Results`-Objekt gespeichert ist.

Boxen werden nur innerhalb desselben Klassennamens fusioniert. Eine Klasse, die nur ein Mitglied kennt, wird unverändert weitergegeben und dafür nicht benachteiligt. Die Bewertungsskalierung verwendet einen klassenweisen Nenner, sodass die Bewertung einer nur einzeln bekannten Klasse erhalten bleibt.

Bei teilweiser Überschneidung wird eine Warnung mit den Klassen protokolliert, die nicht allen Mitgliedern gemeinsam sind. Lies diese Warnung sorgfältig. Ein Checkpoint mit Platzhalternamen wie `class_0` erzeugt eine Vereinigung, die von allen anderen Mitgliedern getrennt ist. In diesem Fall findet keine Fusion zwischen den Mitgliedern statt.

Gibt ein Mitglied eine Klassen-ID außerhalb seines eigenen `names` zurück, wird `RuntimeError` ausgelöst.

## Fremde Detektoren

<code-tabs name="external" />

`ExternalDetector(fn, names)` bindet jede aufrufbare Funktion ein, die ein PIL-Bild entgegennimmt und `(boxes, scores, labels)` zurückgibt. Die Boxen müssen als xyxy-Koordinaten in Pixeln des Originalbildes vorliegen. Der Wrapper prüft Anzahl der Rückgabewerte, Boxform, übereinstimmende Längen und das Vorhandensein jeder Klassen-ID in `names`. Den Schwellenwert `conf` wendet er selbst an.

So kann ein Detektor, den LibreYOLO nicht geladen hat, an der Fusion teilnehmen.

## Aufruf

<code-tabs name="sources" />

Die Aufrufsignatur entspricht der eines einzelnen Modells und akzeptiert dieselben Quellen: Bilder, Ordner, Listen, Videos, Bildschirmaufnahmen, Webcams und Netzwerkstreams. Live-Quellen benötigen aus demselben Grund wie an anderer Stelle `stream=True`.

| Argument | Standard | Hinweise |
|---|---|---|
| `conf` | `0.25` | Je Mitglied; ein Skalar gilt für alle oder ein Wert je Mitglied |
| `iou` | `0.45` | Eigener NMS-Schwellenwert jedes Mitglieds, nicht der Fusionsschwellenwert |
| `imgsz` | `None` | Eine `list` wird je Mitglied gelesen; ein `int` oder Tupel gilt für alle |
| `device` | `None` | Skalar oder ein Wert je Mitglied, sodass Mitglieder auf unterschiedlichen Geräten liegen können |
| `classes` | `None` | Filtert das fusionierte Ergebnis anhand der Klassen-IDs der Vereinigung |
| `max_det` | `300` | Gilt für das fusionierte Ergebnis |

Da eine `list` bei `imgsz` Werte je Mitglied bezeichnet, bedeutet `imgsz=[480, 640]` eine Größe von 480 für das erste und 640 für das zweite Mitglied. `imgsz=(480, 640)` bezeichnet dagegen eine rechteckige Größe für alle. Diese Unterscheidung wird leicht übersehen.

Die Mitglieder werden unabhängig vom angeforderten Wert mit einem `max_det` von mindestens 300 aufgerufen. So liefert jedes großzügig Ergebnisse und das Ensemble beschneidet sie einmal am Ende.

Das Bild wird einmal decodiert und dasselbe Objekt an jedes Mitglied übergeben. `batch` wird aus Gründen der API-Parität akzeptiert, aber ignoriert. Bilder werden sequenziell verarbeitet.

## Rückgabewert

Zurückgegeben wird ein gewöhnliches `Results`-Objekt desselben Typs wie bei einem einzelnen Modell. `names` enthält den vereinigten Klassenraum. Alles unter [Arbeiten mit Ergebnissen](/docs/predict/results) gilt unverändert.

Der einzige Unterschied ist `result.speed`, das ein Ensemble tatsächlich befüllt. Die Schlüssel heißen `member_0`, `member_1` und so weiter sowie `fusion`, jeweils in Millisekunden. Dies ist die einzige Stelle der Bibliothek, an der `speed` ausgefüllt wird.

Zeilen mit nicht endlichen Boxen oder Bewertungen werden vor der Fusion entfernt. Wenn die Mitglieder auf unterschiedlichen Geräten liegen, wird die Fusion auf dem Gerät des ersten Mitglieds ausgeführt, das ein Ergebnis zurückgegeben hat.

## Einschränkungen eines Ensembles

`val()` und `export()` lösen jeweils `NotImplementedError` aus und verweisen auf die Mitglieder. Validiere und exportiere jedes einzeln. Es gibt überhaupt keine Methode `train`, sodass ihr Aufruf `AttributeError` auslöst.

Halbe Genauigkeit wird nicht auf Ensemble-Ebene gesteuert. `half=True` verwendet denselben gewarnten wirkungslosen Pfad wie an anderer Stelle. Konfiguriere die Genauigkeit für jedes Mitglied.

Für Ensembles gibt es keine Befehlszeilenschnittstelle. Es handelt sich um eine Python-API.
