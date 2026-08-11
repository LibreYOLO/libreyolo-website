---
title: Arbeiten mit Ergebnissen
seo_title: Das Results-Objekt von LibreYOLO
description: >-
  Ein Results-Objekt je Bild mit einem Slot pro Nutzlasttyp: Boxen, Masken,
  Keypoints, Wahrscheinlichkeiten, Tiefe, Panoptik, OCR und mehr. Darstellung,
  Speichern und JSON.
lead: >-
  Jede Vorhersage gibt je Bild ein Results-Objekt zurück. Es besitzt einen
  benannten Slot für jede Art von Nutzlast. Bis auf die vom Modell erzeugten
  Slots sind alle leer. Dieselben Slots stehen bei exportierten Artefakten zur
  Verfügung.
keywords:
  - YOLO Results Objekt Python
  - results.boxes xyxy
  - Results als JSON
  - annotiertes Bild speichern
  - Segmentierungsmasken Python
  - Keypoints Ergebnisse
  - Tiefenkarte Results
  - Results Zusammenfassung
  - ONNX gleiche Results
last_verified: 1.5.0
verification: >-
  Nutzlastklassen, Slots, Verschiebungssemantik, summary(), to_json(), plot(),
  save() und cutout() aus libreyolo/utils/results.py gelesen. Verhalten bei
  Annotation und Schreiben auf den Datenträger aus
  InferenceRunner._save_annotated_image in libreyolo/models/base/inference.py
  und resolve_save_path in libreyolo/utils/general.py. Weiterleitung anhand der
  Dateiendung aus LibreYOLO() in libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Boxen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.orig_shape)   # (Höhe, Breite) des Quellbildes

        print(result.path)         # Quellpfad, None bei einer Eingabe im
        Arbeitsspeicher


        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Normalisierte Koordinaten
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy[:1])    # Pixel, x1 y1 x2 y2

        print(result.boxes.xywh[:1])    # Pixel, Mittelpunkt x, Mittelpunkt y,
        Breite, Höhe

        print(result.boxes.xyxyn[:1])   # dieselbe Box geteilt durch Breite und
        Höhe

        print(result.boxes.xywhn[:1])
    - label: NumPy und Geräte
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(SAMPLE_IMAGE)


        # Jede Methode gibt ein neues Results-Objekt zurück; das Original bleibt
        unverändert.

        as_numpy = result.numpy()

        on_cpu = result.cpu()


        print(type(as_numpy.boxes.xyxy).__name__)

        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary und to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # Derselbe Inhalt als String mit denselben Schlüsselwortargumenten.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Annotierte Bilder
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        # save=True zeichnet die Nutzlast und speichert sie unter
        runs/detect/predict*.

        result = model(SAMPLE_IMAGE, save=True)

        print(result.saved_path)
  exported:
    - label: Export-Extra installieren
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Dasselbe Results-Objekt von einem exportierten Artefakt
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")

        path = model.export(format="onnx")   # gibt den geschriebenen Pfad
        zurück


        # LibreYOLO() leitet anhand der Dateiendung weiter.

        exported = LibreYOLO(path)

        result = exported(SAMPLE_IMAGE)


        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Ein Objekt mit einem Slot je Nutzlast

Eine Vorhersage für ein Bild gibt ein `Results`-Objekt zurück. Es enthält achtzehn Nutzlast-Slots. Ein Modell füllt nur die Slots, die seine Aufgabe erzeugt. Jeder andere Slot ist `None`. `result.masks` bei einem Detektor ergibt daher `None` statt eines Fehlers.

| Slot | Klasse | Form | Erzeugt von |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` sowie Bewertungen und Klassen | Objekterkennung und jede Aufgabe mit vorheriger Lokalisierung |
| `masks` | `Masks` | `(N, H, W)` | Instanzsegmentierung |
| `keypoints` | `Keypoints` | `(N, K, 2)` oder `(N, K, 3)` | Posenschätzung |
| `probs` | `Probs` | `(C,)` | Klassifizierung |
| `obb` | `OBB` | `(N, 7)` oder `(N, 8)` | Orientierte Boxen |
| `gaze` | `Gaze` | `(N, 2)` Neigung und Gierung im Bogenmaß | Blickrichtungsschätzung |
| `points` | `Points` | `(N, 4)` als x, y, Klasse, Konfidenz | Punktlokalisierung |
| `semantic_mask` | `SemanticMask` | `(H, W)` Klassen-IDs | Semantische Segmentierung |
| `panoptic` | `PanopticSegmentation` | `(H, W)` Segment-IDs sowie `segments_info` | Panoptische Segmentierung |
| `depth_map` | `DepthMap` | `(H, W)` Gleitkommazahlen | Tiefenschätzung |
| `normal_map` | `NormalMap` | `(H, W, 3)` Einheitsvektoren | Oberflächennormalen |
| `edges` | `EdgeMap` | `(H, W)` Gleitkommazahlen in `[0, 1]` | Kantenerkennung |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Restauration und Super-Resolution |
| `matte` | `Matte` | `(H, W)` Gleitkommazahlen in `[0, 1]` | Alpha Matting und Hintergrundentfernung |
| `ocr` | `OCRRegions` | `(N, 4, 2)` Polygone sowie Transkriptionen | Texterkennung und -erkennung |
| `embeddings` | `Embeddings` | `(N, D)` L2-normalisierte Zeilen | Aufgabe `embed` |
| `identities` | `Identities` | N Namen und Bewertungen | Aufgabe `embed` mit einer Galerie |
| `meshes` | `Meshes` | Körperparameter und optionale Eckpunkte | Rekonstruktion eines Körper-Meshes |

Daneben stehen Felder, die jedes Ergebnis besitzt: `orig_shape` als `(height, width)`, `path` als Quellpfad oder `None` bei einer Eingabe im Arbeitsspeicher, `names` als Zuordnung von Klassen-ID zu Klassenname, `frame_idx` für Video- und Live-Frames, `track_id` beim Tracking sowie `restore_scale`, der ganzzahlige Hochskalierungsfaktor eines Restaurationsergebnisses.

`result.normals` ist ein Alias für `result.normal_map`.

`result.speed` ist bei jedem Ergebnis vorhanden, wird aber nur von [Ensembles](/docs/predict/ensembling) befüllt. Dort heißen seine Schlüssel `member_0`, `member_1` und `fusion` und enthalten Millisekunden. Bei einem einzelnen Modell bleibt es ein leeres Dictionary.

## Boxen

<code-tabs name="basic" />

`Boxes` speichert Koordinaten und Bewertungen in getrennten Arrays statt in einem gepackten Tensor.

| Attribut | Inhalt |
|---|---|
| `xyxy` | `(N, 4)` absolute Pixel, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` absolute Pixel, Mittelpunkt x, Mittelpunkt y, Breite, Höhe |
| `xyxyn`, `xywhn` | Dieselben Werte geteilt durch Bildbreite und -höhe |
| `conf` | `(N,)` Konfidenz |
| `cls` | `(N,)` Klassen-ID als Gleitkommazahl |
| `id` | `(N,)` Tracking-ID oder `None` |
| `is_track` | Gibt an, ob `id` gesetzt ist |
| `data` | Alles zusammengefügt: Boxen, optionale ID, Konfidenz, Klasse |

`cls` ist ein Gleitkomma-Array. Verwende es daher als `result.names[int(cls)]`.

`xyxyn` und `xywhn` benötigen `orig_shape`, das `Results` für dich ausfüllt.

## Dichte Nutzlasten

Nutzlasten für das gesamte Bild verhalten sich anders als instanzbezogene Nutzlasten. Das ist beim Slicing relevant.

`SemanticMask` enthält Klassen-IDs der Form `(H, W)` auf dem ursprünglichen Canvas. Der Wert `255` ist als Ignorierwert reserviert und wird nie als Klasse gezählt. `classes` führt die vorhandenen IDs ohne diesen Wert auf. `class_mask(id)` gibt ein boolesches Array der Form `(H, W)` zurück.

`PanopticSegmentation` enthält Segment-IDs der Form `(H, W)`, wobei `0` die Void-ID ist. Eine Liste `segments_info` aus Dictionaries enthält mindestens `id` und `category_id`. `segment_ids` führt die vorhandenen IDs auf, `segment_mask(id)` wählt eine davon aus.

`DepthMap` enthält die relative inverse Tiefe der Form `(H, W)`. Höhere Werte bedeuten eine geringere Entfernung und sind keine metrischen Meter. Die Eigenschaften `min`, `max` und `mean` gelten für endliche Werte. `normalized()` skaliert auf `[0, 1]`.

`NormalMap` enthält Einheitsvektoren der Form `(H, W, 3)` im OpenCV-Kamerakoordinatensystem. `+x` zeigt nach rechts, `+y` nach unten und `+z` in die Szene. Eine der Kamera zugewandte Fläche ist daher `(0, 0, -1)`. `assert_normalized()` prüft, ob jedes Pixel endlich ist und Einheitslänge besitzt.

`EdgeMap` enthält float32-Werte der Form `(H, W)` in `[0, 1]`. Die kontinuierliche Karte bleibt ohne Schwellenwert erhalten. Mit `binary(threshold=0.5)` legst du den Grenzwert fest.

`Matte` enthält float32-Werte der Form `(H, W)` in `[0, 1]`, wobei `1` vollständig zum Vordergrund gehört. `array` gibt die auf diesen Bereich begrenzten Werte als float32 zurück.

`RestoredImage` enthält ein uint8-RGB-Bild der Form `(H, W, 3)`. `array` stellt das rohe ndarray bereit und `save(path)` schreibt es.

`Probs` enthält einen Wahrscheinlichkeitsvektor für das Bild. `top1` und `top5` sind Klassenindizes, `top1conf` und `top5conf` die entsprechenden Bewertungen.

`Embeddings` enthält bereits L2-normalisierte Zeilen der Form `(N, D)`, sodass die Kosinusähnlichkeit einem Skalarprodukt entspricht. `similarity(other)` gibt gegenüber einer Galerie `(N, M)` und gegenüber einem einzelnen Vektor `(N,)` zurück. `verify(i, j, threshold=0.4)` vergleicht zwei Zeilen.

`OCRRegions` enthält Polygone der Form `(N, 4, 2)` in Lesereihenfolge. Die Eckpunkte sind oben links, oben rechts, unten rechts und unten links angeordnet. Transkriptionen stehen in `texts`, Erkennungsbewertungen in `conf` und Detektionsbewertungen in `det_conf`. Da es sich um echte gedrehte Polygone handelt, befüllen sie `boxes` nicht. `ocr.xyxy` liefert achsenparallele Hüllen, wenn du Rechtecke benötigst.

## Slicing und Verschieben

`result[i]` gibt ein neues `Results`-Objekt mit einer Instanz zurück. Instanzbezogene Nutzlasten werden geschnitten, Nutzlasten für das gesamte Bild unverändert übernommen. Das Slicing eines Klassifizierungsergebnisses kann seinen Wahrscheinlichkeitsvektor daher nicht auf eine Klasse verkürzen und das Slicing eines Tiefenergebnisses kann die Anordnung `(H, W)` nicht beschädigen.

`len(result)` zählt Instanzen: Boxen, Punkte, Embeddings, OCR-Regionen oder Meshes. Jede dichte Nutzlast für das gesamte Bild zählt als `1`. Ein Ergebnis ohne Inhalt hat die Länge `0`.

`to()`, `cpu()`, `cuda()` und `numpy()` geben jeweils ein neues `Results`-Objekt zurück, in dem alle befüllten Slots umgewandelt wurden. Das Original wird nicht verändert.

`update()` ist die einzige Methode, die direkt verändert. Sie ersetzt benannte Slots und gibt dasselbe Objekt zurück.

## JSON

<code-tabs name="json" />

`summary()` gibt eine Liste einfacher Dictionaries zurück. `to_json()` übergibt diese Liste an `json.dumps`. Beide akzeptieren dieselben drei Argumente: `normalize=False` schaltet Koordinaten auf `[0, 1]` um, `decimals=5` legt die Rundung fest und `embeddings=False` steuert, ob Embedding-Vektoren enthalten sind.

Die Form einer Zeile richtet sich nach der Nutzlast. Erkennungszeilen enthalten `name`, `class`, `confidence` und ein Dictionary `box`. Bei vorhandenen Masken kommt `segments` hinzu, bei orientierten Boxen `obb` und `corners`, bei Blickrichtungen Winkel in Bogenmaß und Grad unter `gaze`, beim Tracking `track_id` und bei vorhandenen Meshes Parameter unter `mesh`.

Wenn keine Boxen vorhanden sind, bestimmt eine Nutzlast die Zeilen. OCR erzeugt eine Zeile je Region mit `text`, Punkte eine Zeile je Punkt, Panoptik eine Zeile je Segment mit `pixel_count` und `pixel_fraction`, Semantik eine Zeile je vorhandener Klasse und Klassifizierung die fünf besten Klassen. Tiefe, Normalen, Kanten, Restauration und Matting erzeugen jeweils eine einzelne Zusammenfassungszeile, die die Karte statt ihrer Pixel beschreibt.

Zwei Nutzlasten werden bewusst gekürzt. Ein Embedding-Vektor wird nur als `embedding_dim` gemeldet, da eine Zeile mit 512 Gleitkommazahlen etwa 2 KB je Gesicht umfasst. Übergib `embeddings=True`, um die Werte einzuschließen. Mesh-Eckpunkte werden nie aufgenommen, da dies Zehntausende Koordinaten je Person wären. Verwende für die Geometrie `result.meshes.vertices` oder `result.meshes.save_obj(path)`.

## Zeichnen und Speichern

<code-tabs name="saving" />

`predict(save=True)` annotiert und schreibt die Ausgabe. Die Zeichenroutine wird anhand des befüllten Slots ausgewählt. Ein semantisches Ergebnis wird als farbige Maske geschrieben, ein Tiefenergebnis als Tiefenvisualisierung, ein panoptisches Ergebnis mit seinen Segmenten, eine Matte als RGBA-PNG mit transparentem Hintergrund und eine Detektorausgabe als Boxen mit darunterliegenden Masken. Der geschriebene Pfad wird als `result.saved_path` an das Ergebnis angehängt.

`Results.plot()` ist enger gefasst, als der Name vermuten lässt. Die Methode ist nur für Normalen- und Kantenkarten definiert und löst bei allen anderen Nutzlasten `NotImplementedError` aus. Verwende für andere Aufgaben `save=True`.

`Results.save(path)` ist ebenfalls eng gefasst. Die Methode schreibt ein Matte-Ergebnis als RGBA-PNG-Ausschnitt mit transparentem Hintergrund und löst sonst `NotImplementedError` aus. `Results.cutout()` gibt dasselbe RGBA-Array zurück, ohne es zu schreiben. Beide benötigen das Quellbild aus `result.path` oder als Argument `image=`.

Zwei Nutzlasten besitzen eigene Schreibmethoden: `result.restored.save(path)` für ein restauriertes Bild und `result.meshes.save_obj(path, index=0)` für ein Mesh.

Informationen zu den Speicherorten sowie zum Verhalten von `output_path` und `output_file_format` findest du unter [Vorhersagequellen](/docs/predict/sources).

## Gleiche Objekte aus exportierten Artefakten

<code-tabs name="exported" />

`LibreYOLO()` leitet anhand der Dateiendung weiter. Ein exportiertes Artefakt wird daher mit demselben Aufruf wie ein `.pt`-Checkpoint geladen und gibt dasselbe `Results`-Objekt zurück. `.onnx`-, `.engine`-, `.pte`- und `.mnn`-Dateien werden anhand ihrer Endung erkannt. Gleiches gilt für OpenVINO-, Paddle- und ncnn-Verzeichnisse sowie eine Triton-Modell-URL. Code, der `result.boxes.xyxy` liest, ändert sich beim Austausch eines Modells gegen seine exportierte Variante nicht. Unter [Export](/docs/export) findest du alle Formate.

Wenn du stattdessen direkt die API der Laufzeitumgebung verwendest, bist du selbst für Vorverarbeitung, Nachverarbeitung und Klassennamen verantwortlich.
