---
title: Results-Typen
seo_title: Referenz zum LibreYOLO-Results-Objekt
description: >-
  Jede Payload, die ein LibreYOLO-Results-Objekt enthalten kann, mit einem Slot
  pro Aufgabenform: Boxen, Masken, Keypoints, Probs, OBB, Tiefe, OCR, Embeddings
  und zehn weitere.
lead: >-
  Results ist der einzelne Rückgabetyp pro Bild für jedes LibreYOLO-Modell. Er
  enthält 18 optionale Payload-Slots, einen pro Aufgabenform, und füllt nur
  diejenigen, die das Modell erzeugt hat.
keywords:
  - libreyolo results objekt
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - libreyolo results to_json
last_verified: 1.5.0
verification: >-
  Slot-Namen, Formen, Eigenschaften und Standardwerte aus
  libreyolo/utils/results.py für v1.5.0. Semantik aus den Docstrings der
  Payload-Klassen.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Jede Payload wird gemeinsam verschoben.
        result = result.cpu().numpy()

        # Zeilen als einfache Dictionaries und anschließend als JSON.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## Das Results-Objekt

Ein `Results` beschreibt ein Bild. Eine einzelne Bildquelle gibt eines davon
zurück, eine Listenquelle oder ein Verzeichnis eine Liste. `stream=True` gibt
einen Generator zurück, der solche Objekte liefert.

| Attribut | Typ | Bedeutung |
|---|---|---|
| `orig_shape` | `(int, int)` | Höhe und Breite des Originalbilds |
| `path` | `str` | Quellpfad, wenn die Eingabe vom Datenträger stammt |
| `names` | `dict[int, str]` | Zuordnung von Klassenindex zu Klassenname |
| `speed` | `dict[str, float]` | Millisekunden pro Verarbeitungsstufe |
| `track_id` | Tensor | Track-IDs, wenn das Ergebnis von `track()` stammt |
| `frame_idx` | `int` | Frame-Index für Video- und Streamquellen |
| `restore_scale` | `int` | Vergrößerungsfaktor von Ausgabe zu Eingabe bei einem Restaurierungsergebnis, sonst `1` |

<code-tabs name="usage" />

## Payload-Slots

Jeder Slot ist `None`, sofern das Modell ihn nicht erzeugt hat. Die Aufgabe
einer Familie bestimmt den gefüllten Slot.

| Slot | Klasse | Aufgabe |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, mit Galerie |
| `meshes` | `Meshes` | mesh |

`result.normals` ist ein Lese- und Schreibalias für `result.normal_map`.

Mehr als ein Slot kann gleichzeitig gesetzt sein. Ein Segmentierungsmodell
füllt sowohl `boxes` als auch `masks`. Ein Blickmodell füllt `boxes` mit den
Gesichtsboxen und `gaze` mit den Winkeln. Ein Mesh-Modell füllt `boxes` mit den
Personenboxen und richtet die Zeilen von `meshes` daran aus.

## Boxes

Erkennungsboxen für ein Bild.

| Element | Rückgabe |
|---|---|
| `xyxy` | Eckkoordinaten in Pixeln des Originalbilds |
| `xywh` | Mittelpunkt und Größe in Pixeln |
| `xyxyn` | Auf `[0, 1]` normalisierte Ecken |
| `xywhn` | Auf `[0, 1]` normalisierter Mittelpunkt und Größe |
| `conf` | Confidence pro Box |
| `cls` | Klassenindex pro Box |
| `id` | Track-ID pro Box oder `None` |
| `is_track` | `True`, wenn Track-IDs vorhanden sind |
| `data` | Der gepackte Tensor |

`with_id(id)` und `with_orig_shape(orig_shape)` geben ein neues `Boxes` zurück,
in dem das jeweilige Feld ersetzt ist.

## Masks

Instanzmasken für ein Bild. `data` ist der Maskentensor. `xy` gibt Konturen pro
Instanz in Pixeln zurück, `xyn` die normalisierten Konturen.

## Keypoints

Pose-Keypoints, deren Zeilen an `boxes` ausgerichtet sind. `xy` enthält das
Koordinatenpaar pro Keypoint, `xyn` das normalisierte Paar. `conf` ist der
dritte Kanal, falls die Daten einen enthalten, andernfalls `None`.
`has_visible` ist ein boolesches Array. Es ist wahr, wo `conf > 0`, und
vollständig wahr, wenn kein Confidence-Kanal vorhanden ist.

## Points

Punktlokalisierung für ein Bild. `data` besitzt die Form `(N, 4)` mit den
Zeilenwerten `x, y, class, confidence`. Die Koordinaten sind absolute Pixel.
`xy`, `cls` und `conf` teilen die Spalten auf. `xyn` normalisiert die
Koordinaten.

## Probs

Klassifikations-Scores. `top1` ist der Index mit dem höchsten Score, `top5`
enthält die fünf besten Indizes. `top1conf` und `top5conf` geben ihre Scores
zurück.

## OBB

Orientierte Boxen. `data` enthält pro Zeile 7 oder 8 Werte: `xywhr`, eine
optionale Track-ID und anschließend Confidence und Klasse.

| Element | Rückgabe |
|---|---|
| `xywhr` | Mittelpunkt, Größe und Rotation in Radiant |
| `xyxyxyxy` | Die vier Ecken in Pixeln |
| `xyxyxyxyn` | Die vier normalisierten Ecken |
| `xyxy` | Achsenparallele Hülle in Pixeln |
| `conf`, `cls`, `id`, `is_track` | Wie bei `Boxes` |

## Gaze

Blickwinkel pro Gesicht in Radiant mit der Form `(N, 2)`. Die Zeilen sind an
den Gesichtsboxen in `boxes` ausgerichtet. Spalte 0 ist Pitch und Spalte 1 Yaw
nach der L2CS-Konvention. Positives Yaw dreht den Blick nach links aus Sicht
der Person, positiver Pitch nach unten. `pitch_deg` und `yaw_deg` rechnen in
Grad um. `direction_3d` gibt den Einheitsrichtungsvektor zurück.

## SemanticMask

Dichte semantische Karte mit der Form `(H, W)` aus ganzzahligen Klassen-IDs auf
der Canvas des Originalbilds. `255` ist der Ignore-Wert und zählt nie als
Klasse (`SemanticMask.IGNORE_INDEX`). `classes` führt die vorhandenen
Klassen-IDs auf. `class_mask(class_id)` gibt die boolesche Maske einer Klasse
zurück.

## PanopticSegmentation

Jedes Pixel erhält genau ein nicht überlappendes Segment. Dadurch werden
Stuff-Regionen und Thing-Instanzen vereint. `data` ist eine ganzzahlige
Segment-ID-Karte der Form `(H, W)`. Segment-ID `0` ist nicht beschriftet
(`PanopticSegmentation.IGNORE_INDEX`). `segments_info` ist eine Liste von
Dictionaries mit einem Eintrag pro Segment und mindestens
`{"id": int, "category_id": int}`. `id` entspricht einem Wert in der Karte,
`category_id` indiziert `names`. `segment_ids` führt die vorhandenen IDs auf.
`segment_mask(segment_id)` gibt die boolesche Maske eines Segments zurück.

Die Unterscheidung zwischen Thing und Stuff ist eine Eigenschaft der Kategorie
und nicht des Segments. Eine Payload kann sie als `"isthing": bool` auf jedes
Segment denormalisieren. In diesem Fall muss der Wert mit der Zuordnung auf
Kategorieebene übereinstimmen.

## DepthMap

Dichte Karte relativer inverser Tiefe mit der Form `(H, W)` aus Float-Werten
auf der Canvas des Originalbilds. Höhere Werte liegen näher an der Kamera. Die
Werte sind relativ und keine metrischen Meterangaben. `min`, `max` und `mean`
werden über endliche Werte berechnet. `normalized()` skaliert die Karte auf
`[0, 1]`.

## NormalMap

Dichtes Oberflächennormalenfeld als float32 mit der Form `(H, W, 3)` auf der
Canvas des Originalbilds im OpenCV-Kamerarahmen: `+x` nach rechts, `+y` nach
unten und `+z` in die Szene. Normalen weisen zur Kamera. Eine frontoparallele
Oberfläche entspricht daher `(0, 0, -1)`. Jedes Pixel ist ein Einheitsvektor.
`assert_normalized(atol=1e-4)` prüft diese Invariante.

## EdgeMap

Dichte Kantenwahrscheinlichkeitskarte als float32 mit der Form `(H, W)` auf der
Canvas des Originalbilds. `0` steht für keine Kante, `1` für eine Kante. Die
kontinuierliche Karte bleibt erhalten, damit der aufrufende Code den
Schwellenwert wählen kann. `binary(threshold=0.5)` wendet einen an. `array`
gibt die numpy-Ansicht zurück.

## RestoredImage

Das restaurierte RGB-Bild als uint8 mit der Form `(H, W, 3)`. Bei
Super-Resolution ist die Canvas `Results.restore_scale`-mal so groß wie die
Eingabe. `array` gibt die numpy-Ansicht zurück. `save(path)` schreibt das Bild.

## Matte

Weiches Deckkraft-Matte als float32 mit der Form `(H, W)` im Bereich `[0, 1]`
auf der Canvas des Originalbilds. `1` ist vollständig Vordergrund, `0`
vollständig Hintergrund. Ein weiches Matte umfasst eine harte Maske zur
Hintergrundentfernung bei Schwellenwert 0.5 und behält die geglätteten Kanten,
die eine binäre Maske verwirft. `array` gibt die numpy-Ansicht zurück.

Bei einem Matte-Ergebnis gibt `Results.cutout(image=None)` ein RGBA-uint8-Array
der Form `(H, W, 4)` zurück, dessen vierter Kanal das Matte ist.
`Results.save(path, image=None)` schreibt diesen Ausschnitt als PNG mit
transparentem Hintergrund. Beide verwenden RGB aus `image`, falls angegeben,
und laden es andernfalls aus `Results.path`.

## OCRRegions

Lokalisierter Text mit Transkripten. `data` enthält Float-Polygone der Form
`(N, 4, 2)` in Pixeln des Originalbilds. Die Punkte sind oben links, oben
rechts, unten rechts und unten links angeordnet. Regionen stehen in
Lesereihenfolge, von oben nach unten und anschließend von links nach rechts.
`texts` ist die Liste der N Transkripte. `conf` ist der Recognition-Score pro
Region, `det_conf` der Erkennungs-Score, beide mit der Form `(N,)`.

Erkennungsvierecke sind echte Polygone und füllen daher `Results.boxes` nicht.
`xyxy` gibt ihre achsenparallelen Hüllen zurück.

## Embeddings

L2-normalisierte Vektoren der Aufgabe `embed`, immer mit der Form `(N, D)`.
Ein Ergebnis für das ganze Bild enthält eine Zeile und keine Boxen.
Regions-Embeddings sind zeilenweise an `boxes` ausgerichtet. Da jede Zeile
normalisiert ist, entspricht die Kosinusähnlichkeit einem Skalarprodukt.

| Element | Rückgabe |
|---|---|
| `dim` | `D` |
| `normalized` | Die erneut normalisierten Zeilen |
| `similarity(other)` | Paarweise Kosinusähnlichkeit mit einem anderen `Embeddings` oder Tensor |
| `verify(i, j, threshold=0.4)` | `True`, wenn die Zeilen `i` und `j` übereinstimmen |

## Identities

Benannte Galerieübereinstimmungen, deren Zeilen an `embeddings` ausgerichtet
sind. Sie werden erzeugt, wenn einer Vorhersage der Aufgabe `embed` eine
`Gallery` übergeben wird. `name` ist eine Liste, deren Eintrag unter dem
Übereinstimmungsschwellenwert `None` ist. Der nächstgelegene Name unterhalb des
Schwellenwerts wird nie geraten. `score` ist das Array der Übereinstimmungswerte.
`data` paart beide.

## Meshes

Parametrische menschliche Körpernetze, deren Zeilen an den Personenboxen in
`boxes` ausgerichtet sind. Alle Werte liegen im Kamerarahmen des Originalbilds.
`transl` ist metrisch in Metern, wobei `+z` von der Kamera weg weist.
`vertices` und `joints3d` sind metrisch und enthalten `transl` bereits.
`joints2d` liegt in Pixeln auf der Canvas des Originalbilds und nicht auf dem
vom Netz gesehenen Ausschnitt. Kein Feld enthält einen Welt- oder
Schwerkraftrahmen.

Parameterlayouts unterscheiden sich zwischen Körpermodellen. Formen sind daher
nicht fest codiert. `body_model` benennt die Parametrisierung. Anzahlen werden
aus den Tensoren zurückgelesen: `num_vertices`, `num_joints`, `num_betas` und
`has_vertices`. `params` gibt das Parameter-Dictionary zurück.
`save_obj(path, index=0)` schreibt ein Mesh. Die Felder heißen
`global_orient`, `body_pose`, `betas`, `transl`, `vertices`, `faces`,
`joints3d`, `joints2d`, `conf`, `focal_length` und `extras`.

Bei `body_model="mhr"` sind Rotationen Euler-Winkel in Radiant statt
Axis-Angle. `body_pose` ist ein flacher Parametervektor pro Gelenk und kein
Tripel pro Gelenk. `betas` sind Identity-Blendshape-Koeffizienten.
Skelettskalierung, Handpose und Gesichtsausdruck liegen in `extras`.

## Konvertierung und Auswahl

Jede Payload stellt `to(*args, **kwargs)`, `cpu()`, `cuda()` und `numpy()`
bereit. Ein Aufruf am `Results`-Objekt wendet die Methode auf alle gefüllten
Slots gleichzeitig an.

<code-tabs name="convert" />

`result[idx]` wählt Zeilen über alle zeilenweise ausgerichteten Payloads aus.
`len(result)` ist die Anzahl der Erkennungen oder, wenn keine Boxen vorhanden
sind, der Punkte. `result.update(...)` gibt eine Kopie zurück, in der die
benannten Slots ersetzt sind. Die Methode akzeptiert jeden Slot sowie
`track_id` und `restore_scale`.

## summary und to_json

`summary(normalize=False, decimals=5, embeddings=False)` gibt je nach gefüllten
Slots eine Liste einfacher Dictionaries mit einer Zeile pro Erkennung,
Segment, Punkt oder Region zurück. `to_json(**kwargs)` leitet seine Argumente
an `summary` weiter und gibt den JSON-String zurück.

`plot()` rendert ein dichtes Normalen- oder Kantenergebnis in seiner
kanonischen Visualisierung. Bei anderen Ergebnistypen löst es einen Fehler aus.
Annotierte Bilder der übrigen Aufgaben werden durch `predict(save=True)`
erzeugt.

