---
title: Datensatzformate
seo_title: LibreYOLO-Datensatzformate für jede Aufgabe
description: >-
  Datensatzdateivertrag je kanonischer Aufgabe: YAML-Schlüssel,
  Ordnerstrukturen, Labelzeilen, Masken- und Kartenkonventionen sowie der
  jeweilige Loader.
lead: >-
  Diese Seite entspricht dem Datensatzdateivertrag in docs/dataset_schema.md der
  Bibliothek. Sie beschreibt die YAML-Schlüssel und die Struktur auf dem
  Datenträger, die jede kanonische Aufgabe erwartet.
keywords:
  - LibreYOLO Datensatzformat
  - YOLO Labelformat
  - data.yaml
  - Segmentierungsmaske Datensatz
  - COCO Panoptic Format
  - Tiefendatensatz
  - Pose kpt_shape
last_verified: 1.5.0
verification: >-
  Entspricht docs/dataset_schema.md im LibreYOLO-Repository in v1.5.0.
  Loader-Namen wurden mit libreyolo/data/ abgeglichen.
snippets:
  usage:
    - label: Eine Erkennungs-Labelzeile analysieren
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h, auf [0, 1] normalisiert

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) in Pixeln

        print(row)
source_hash: a8282c079624044d
---

## Gemeinsame YAML-Struktur

Gilt für `detect`, `segment`, `pose` und `obb`.

| Schlüssel | Erforderlich | Bedeutung |
|---|---|---|
| `path` | | Datensatzstammverzeichnis |
| `train` | Für Training | Trainingsbilder |
| `val` | Für Validierung | Validierungsbilder |
| `test` | | Testbilder |
| `names` | Ja | Klassenliste oder Zuordnung mit ganzzahligen Schlüsseln |
| `nc` | | Klassenanzahl; muss bei Angabe mit `names` übereinstimmen |
| `download` | | Downloadanweisungen; Python-Skripte benötigen eine ausdrückliche Aktivierung |
| `annotations` | | Zuordnung von Split zu nativer COCO-JSON-Datei für detect, segment und obb |

`train`, `val` und `test` können Bildverzeichnisse, `.txt`-Dateien mit Bildlisten oder Listen daraus sein. Labelpfade folgen einer Ersetzung:

```text
images/.../image.jpg -> labels/.../image.txt
```

Bei einem Datensatz im nativen COCO-JSON-Format ordnet `annotations` einen Split seiner JSON-Datei zu. Der Splitpfad gibt das Bildstammverzeichnis an:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Wenn `names` vorhanden ist, müssen die Kategorienamen im nativen COCO-JSON mit den Klassennamen in der YAML-Datei übereinstimmen. Diese Namen definieren die Modell-Label-IDs. Ohne `names` werden COCO-Kategorie-IDs sortiert und dicht auf `0..N-1` abgebildet.

Eine Datensatz-YAML-Datei enthält keinen Schlüssel `task`. Die ausdrückliche Auswahl von Modell und Aufgabe hat Vorrang.

Für jede Textlabeldatei gelten dieselben Regeln:

- eine `.txt`-Labeldatei je Bild;
- eine fehlende oder leere Labeldatei bedeutet, dass keine Objekte vorhanden sind;
- `class_id` ist eine Ganzzahl in `0..nc-1`;
- Koordinaten sind endliche, auf `[0, 1]` normalisierte Gleitkommazahlen;
- Koordinaten beziehen sich auf Breite und Höhe des Originalbildes;
- Zeilen enthalten weder Konfidenz noch Tracking-ID.

<code-tabs name="usage" />

## detect

Exakt fünf Felder je Zeile:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` ist eine normalisierte achsenparallele Box. `w` und `h` müssen positiv sein.

## segment

Eine Polygonzeile:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` beträgt mindestens 3. Die Koordinatenanzahl nach `class_id` muss gerade sein und das Polygon darf nicht degeneriert sein. Eine Erkennungszeile mit fünf Feldern wird ebenfalls akzeptiert und stellt ein rechteckiges Segment dar.

## pose

Die YAML-Datei ergänzt das erforderliche `kpt_shape` als `[K, 2]` oder `[K, 3]` sowie das optionale `flip_idx`, eine Ganzzahlpermutation von `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Die Feldanzahl beträgt exakt `5 + K * D`, wobei `D` der zweite Wert von `kpt_shape` ist. Keypoint-Koordinaten sind normalisiert. Der optionale Sichtbarkeitswert `v` lautet `0`, `1` oder `2`.

## obb

Exakt neun Felder:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Die vier Punkte sind normalisierte Bildkoordinaten in `[0, 1]` und bilden ein nicht degeneriertes orientiertes Rechteck. Die Labeldatei speichert keinen Winkel.

Der kanonische Parser ist standardmäßig streng und lehnt Koordinaten außerhalb des Bereichs ab. Bei der Aufnahme für Datensatz und Validierung dürfen Koordinaten anderweitig gültiger Labels an Beschnittgrenzen zunächst auf `[0, 1]` begrenzt werden. Degenerierte Boxen werden weiterhin abgelehnt. Die Analyse berücksichtigt die Aufgabe. Neun Felder bedeuten nur im Modus `obb` eine OBB. Im Modus `segment` können sie ein Polygon mit vier Punkten darstellen.

Intern werden normalisierte Ecken in die kanonische Form `xywhr` umgewandelt. Der Winkel im Bogenmaß beschreibt die Drehung der Breitseite um den Boxmittelpunkt. Öffentliche Ergebnisse stellen OBB-Erkennungen als Zeilen `xywhr, conf, cls` bereit.

Beim Laden nativer COCO-JSON-OBBs werden Annotationen in folgender Reihenfolge akzeptiert: `obb` als acht Ecken im Pixelraum; `obb` als `[cx, cy, w, h, angle]` mit Winkel im Bogenmaß; ein COCO-`segmentation`-Polygon oder RLE, das an ein flächenkleinstes Rechteck angepasst wird; sowie eine COCO-`bbox`, die achsenparallel gelesen und kanonisiert wird.

Mosaic und MixUp bleiben beim OBB-Training deaktiviert, bis eine eckenbewusste OBB-Augmentation vorhanden ist.

Der kanonische Zeilenparser heißt `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Jedes Bild wird statt einer `.txt`-Datei mit einer dichten einkanaligen Maske in einem verlustfreien Format, üblicherweise PNG, gekoppelt:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

Die Maske besitzt einen Kanal. PNG-Dateien im Palettenmodus werden als Palettenindizes gelesen. Jeder Pixelwert ist eine Klassen-ID in `0..nc-1`. Der Pixelwert `255` bedeutet Ignorieren und wird von Loss und Metriken ausgeschlossen. Die Maskenauflösung muss der Bildauflösung entsprechen.

Zwei optionale YAML-Schlüssel ergänzen den gemeinsamen Vertrag. `masks_dir` ist der Name des Maskenverzeichnisses, das in jedem Bildpfad `images` ersetzt. Der Standard lautet `masks`. `label_mapping` ist eine Zuordnung `{source_id: train_id}`, die beim Laden auf Maskenpixelwerte angewendet wird. Nicht zugeordnete Quellwerte werden zum Ignorierwert. Trainings-IDs müssen in `0..nc-1` liegen.

Wenn `masks_dir` fehlt, werden die Masken beim Laden aus `segment`-Polygonlabels gerastert, die über die Konvention von `images` zu `labels` aufgelöst werden. Nach den Objektklassen wird eine Klasse `background` angehängt, weshalb `nc` um eins wächst.

Kanonischer Loader: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO übernimmt das COCO-Panoptic-Format unverändert (Kirillov et al., CVPR 2019). Es gibt kein eigenes panoptisches LibreYOLO-Format.

Eine RGB-PNG-Datei je Bild in dessen Auflösung codiert die Segment-ID jedes Pixels in ihrer Farbe:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Jedes Pixel gehört zu genau einem Segment und Segmente überlappen sich nie. Segment-ID `0`, also Schwarz in RGB, bedeutet Void. Solche nicht beschrifteten Pixel sind von der Metrik ausgeschlossen.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` benennt die Segment-ID-PNG-Datei innerhalb von `panoptic_dir`. `segments_info[].id` entspricht einem Wert in dieser PNG-Datei. `iscrowd` markiert Gruppenregionen. Sie sind nie False Negatives. Eine Vorhersage, die eine solche Region größtenteils abdeckt, zählt nicht als False Positive.

Die Unterscheidung zwischen Thing und Stuff ist eine Kategorieeigenschaft. `isthing` steht unter `categories` und nie unter `segments_info`.

COCO-Panoptic-Werte `category_id` sind rohe Datensatz-IDs und üblicherweise nicht zusammenhängend. Modelle sagen zusammenhängende IDs `0..nc-1` vorher. Rohe IDs werden daher anhand des Kategorienamens über `names` in der YAML-Datei abgebildet. Dieselbe Regel verwendet der native COCO-JSON-Loader für die Objekterkennung. Fehlt eine JSON-Kategorie in `names`, wird ein Fehler ausgelöst, statt sie unbemerkt zu verwerfen. Andernfalls würde sie dauerhaft als False Negative bewertet.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` und `panoptic_dir` akzeptieren entweder einen einzelnen Pfad oder eine Zuordnung je Split.

Die Validierung meldet Panoptic Quality, berechnet in Ground-Truth-Auflösung und gemittelt über die vorkommenden Kategorien. Anschließend wird sie in `PQ_things` und `PQ_stuff` aufgeteilt. Die Zuordnung ist eindeutig. Ein vorhergesagtes Segment und ein Ground-Truth-Segment derselben Kategorie stimmen bei einer IoU über 0,5 überein.

Kanonischer Loader: `libreyolo.data.PanopticDataset`.

## depth

Jedes Bild wird mit einer dichten einkanaligen Tiefenkarte gekoppelt:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

Die Karte ist eine einkanalige PNG- oder TIF-Datei oder eine `.npy`-Datei in Bildauflösung. Werte geben die gewöhnliche Tiefe in einer innerhalb des Datensatzes einheitlichen Einheit an. Null, negative, NaN- und unendliche Werte markieren ungültige Pixel und werden von Loss und Metriken ausgeschlossen.

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `depths_dir` | `depths` | Tiefenverzeichnis, das `images` ersetzt |
| `depth_stem_suffix` | | An den Bildstamm angehängtes Suffix; bei Auslassung werden derselbe Stamm und ein Suffix `_depth` versucht |
| `depth_mask_suffix` | `_mask` | Suffix einer Gültigkeitsmaske; Maskenwerte kleiner oder gleich null, NaN und unendlich machen das Tiefenpixel ungültig |
| `depth_scale` | `256.0` | Divisor für ganzzahlig typisierte Tiefenkarten, entsprechend der üblichen 16-Bit-PNG-Konvention |

Gleitkommakarten im `.npy`-Format werden unverändert verwendet und wenden `depth_scale` nicht an.

Kanonischer Loader: `libreyolo.data.DepthDataset`.

## edge

Jedes RGB-Bild wird mit einer einkanaligen, verlustfreien Karte gleichen Stamms und einer optionalen Gültigkeitsmaske gekoppelt:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

Die Karte ist eine einkanalige PNG- oder TIF-Datei und keine RGB-Visualisierung. Ihre Auflösung entspricht dem Bild. Ganzzahlige Karten werden durch den Maximalwert ihres dtype geteilt. Gleitkommakarten müssen bereits endlich sein und in `[0, 1]` liegen. `0` bedeutet keine Kante, `1` eine Kante. Optionale Maskenpixel sind gültig, wenn sie ungleich null sind. Beim Skalieren wird Nächster-Nachbar-Interpolation für Ziele und Masken verwendet. Aufgefüllte Pixel sind ungültig und tragen nicht zur Validierung bei.

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `edges_dir` | `edges` | Kantenkartenverzeichnis, das `images` ersetzt |
| `edge_stem_suffix` | | An den Bildstamm angehängtes Suffix |
| `edge_extension` | `.png` | Verlustfreie Zielendung |
| `edge_invert` | | Auf wahr setzen, wenn Quellkarten schwarze Kanten auf Weiß speichern |
| `masks_dir` | `masks` | Optionales Verzeichnis für Gültigkeitsmasken |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Die Validierung verdünnt kontinuierliche Vorhersagen mit Gradienten-Non-Maximum-Suppression in vier Richtungen und meldet ODS- und OIS-F-Maße über einen konfigurierbaren Schwellenwertdurchlauf. Vorhergesagte und tatsächliche Pixel werden innerhalb von `edge_max_dist * image_diagonal` eins zu eins zugeordnet. Die normalisierte Standardtoleranz beträgt `0.0075`.

Kanonischer Loader: `libreyolo.data.EdgeDataset`. Der Loader stellt nur das Format bereit. Er lädt keine Benchmark-Daten herunter und verteilt sie nicht weiter.

## normal

Jedes Bild wird mit einer dreikanaligen 16-Bit-PNG-Datei gleichen Stamms und einer optionalen Gültigkeitsmaske gleichen Stamms gekoppelt:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

Die PNG-Datei besitzt in Bildauflösung exakt drei als RGB gespeicherte `uint16`-Kanäle. Decodiere sie mit `n = png / 65535 * 2 - 1` und normalisiere danach jeden Vektor erneut. Decodierte Vektoren verwenden das OpenCV-Kamerakoordinatensystem: `+x` zeigt nach rechts, `+y` nach unten, `+z` in die Szene. Sie sind zur Kamera gerichtet. Die optionale Maske ist eine einkanalige PNG-Datei, in der ungleich null gültig bedeutet. Ohne Maske ist jeder endliche, von null verschiedene decodierte Vektor gültig. Ungültige und aufgefüllte Zielpixel werden intern als `(0, 0, 0)` dargestellt. Beim Skalieren werden die drei Komponenten bilinear interpoliert und anschließend erneut normalisiert. Gültigkeitsmasken verwenden Nächster-Nachbar-Interpolation. Eine horizontale Spiegelung negiert außerdem die x-Komponente.

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `normals_dir` | `normals` | Verzeichnis der Normalenkarten, das `images` ersetzt |
| `masks_dir` | `masks` | Optionales Verzeichnis für Gültigkeitsmasken |

Die Validierung meldet mittleren und medianen Winkelfehler in Grad sowie den Anteil gültiger Pixel innerhalb von 11,25, 22,5 und 30 Grad.

Kanonischer Loader: `libreyolo.data.NormalDataset`.

## restore

Jedes beeinträchtigte Eingabebild wird mit einem unverfälschten RGB-Ziel gekoppelt:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Eingabe und Ziel sind RGB-kompatible Bilddateien und ihre Auflösungen müssen exakt übereinstimmen. Die Validierung behält die native Auflösung und füllt nur so weit auf, wie es für das Stapeln eines Batches nötig ist. Metriken werden auf dem ursprünglichen Bild-Canvas berechnet. Beim Training werden ein gekoppelter Beschnitt und eine horizontale Spiegelung auf Eingabe und Ziel angewendet.

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `input_dir` | `inputs` | In Splitpfaden verwendetes Verzeichnis beeinträchtigter Eingaben |
| `target_dir` | `targets` | Verzeichnis unverfälschter Ziele, das `input_dir` ersetzt |
| `target_stem_suffix` | | Vor der Zielsuche an den Eingabestamm angehängtes Suffix |
| `target_stem_suffixes` | | Listenform von `target_stem_suffix` |
| `degradation` | | Metadatenbezeichnung wie `deblur` oder `denoise` |
| `dataset` | | Datensatz- oder Herkunftsbezeichnung |

Klassenartige YAML-Felder sind Schemaplatzhalter. Verwende `nc: 1` und `names: {0: image}`. Restaurationmodelle stellen `Results.restored` statt Erkennungen bereit.

Kanonischer Loader: `libreyolo.data.RestoreDataset`.

## matte

Jedes RGB-Bild wird mit einer einkanaligen Ground-Truth-Matte gleichen Stamms gekoppelt, in der 0 Hintergrund und 255 Vordergrund bedeutet:

```text
images/subject.jpg -> mattes/subject.png
```

Zwei Strukturen werden akzeptiert. Die erste ist ein als `data=` übergebenes Stammverzeichnis mit `images/` und einem automatisch unter `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` oder `alpha/` erkannten Matte-Verzeichnis. Die zweite ist eine YAML-Datei mit `path` sowie `val_images` und `val_mattes` je Split. Optional können `train_images` und `train_mattes` hinzukommen. Jeder Pfad ist relativ zu `path` oder absolut.

Die Matte wird als Graustufen-Opazität in `[0, 1]` gelesen. Bei abweichenden Formen wird sie mit bilinearer Interpolation auf den Vorhersage-Canvas skaliert. Metriken sind MAE und S-Measure (Fan et al., ICCV 2017) auf dem ursprünglichen Bild-Canvas. S-Measure dient als Fitness für die Auswahl des besten Checkpoints.

Klassenartige YAML-Felder sind Schemaplatzhalter. Verwende `nc: 1` und `names: {0: matte}`. Matte-Modelle stellen `Results.matte` bereit.

Die Validierung ist in dieser Version ausschließlich für die Inferenz vorgesehen. Kanonischer Paar-Resolver: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Labels bestehen aus einer JSONL-Datei je Split mit einem JSON-Objekt je Bild:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` ist ein Viereck aus vier Punkten in absoluten Pixelkoordinaten. Die Reihenfolge lautet oben links, oben rechts, unten rechts und unten links. Regionen mit unlesbarem Text verwenden `"text": "###"`, die ICDAR-Konvention für nicht zu berücksichtigende Bereiche. Sie werden von der Erkennungsbewertung ausgeschlossen. Vorhersagen, die sie überlappen, werden bei der Detektionszuordnung ignoriert statt bestraft.

Metriken sind der Detektions-Hmean bei eindeutiger Polygonzuordnung über IoU 0,5, ein End-to-End-F1, der sowohl IoU über 0,5 als auch eine nach NFKC-Normalisierung und Entfernung von Leerraum exakt übereinstimmende Transkription mit Beachtung der Groß- und Kleinschreibung erfordert, sowie 1-NED für zugeordnete Paare. Der End-to-End-F1 ist die Fitness zur Auswahl des besten Checkpoints.

Zwei Strukturen werden akzeptiert: ein als `data=` übergebenes Stammverzeichnis mit `images/<split>/` und `labels/<split>.jsonl` oder eine YAML-Datei mit `path` sowie optionalen Verzeichnisnamen `images` und `labels`.

Klassenartige YAML-Felder sind Schemaplatzhalter. Verwende `nc: 1` und `names: {0: text}`. OCR-Modelle stellen `Results.ocr` bereit.

Die Validierung ist in dieser Version ausschließlich für die Inferenz vorgesehen. Kanonischer Sample-Resolver: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Ein Verzeichnisbaum im ImageFolder-Stil statt Labeldateien:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` ist für das Training erforderlich und definiert die Zuordnung von Klasse zu Index nach sortiertem Ordnernamen. `val/` ist für die Validierung erforderlich. `test/` darf vorhanden sein, wird von den Standardbefehlen für Training und Validierung aber nicht verwendet. Splits außerhalb des Trainings müssen dieselben Klassenordnernamen wie die erwartete Trainings- oder Checkpoint-Klassenmenge enthalten. Unterstützte Bildendungen sind in `libreyolo.data.classify_dataset.IMAGE_EXTENSIONS` definiert.

## gaze und point

Für `gaze` ist kein Datensatzdateivertrag für Training oder Validierung implementiert.

`point` ist eine Modellausgabeaufgabe und kein Datensatzlabelschema. Punktfamilien dürfen vorhandene Labels intern anpassen, beispielsweise Objektmittelpunkte aus Boxzeilen ableiten. Ein reines Textlabelformat für Punkte ist jedoch nicht definiert.
