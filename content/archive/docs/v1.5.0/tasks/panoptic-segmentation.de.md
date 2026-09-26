---
title: Panoptische Segmentierung
seo_title: Panoptische Segmentierung in LibreYOLO
description: >-
  Weise mit LibreYOLO jedem Pixel genau ein Segment zu. Erfahre mehr über die
  unterstützten Familien, das COCO-Panoptic-Datensatzformat sowie die Aufrufe
  für Vorhersage und Validierung.
lead: >-
  Die panoptische Segmentierung weist jedes Pixel genau einem nicht
  überlappenden Segment zu und vereint zählbare Objektinstanzen mit formlosen
  Hintergrundregionen. Der Aufgabenschlüssel lautet panoptic.
keywords:
  - panoptische segmentierung python
  - panoptic quality
  - things und stuff segmentierung
  - COCO panoptic format
  - segment-id-karte
  - PQ metrik
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -panoptic im Dateinamen wählt die Aufgabe aus. Ein task-
        # Argument ist daher nicht nötig.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) Segment-IDs
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Jeweils ein Segment
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # boolesch (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Kleinerer Checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() gibt ein einfaches Dictionary und kein Objekt zurück.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Definition

Die panoptische Segmentierung vereint die beiden anderen
Segmentierungsaufgaben. Jedes Pixel erhält genau ein Segment, Segmente
überlappen sich nie, und ein Segment ist entweder ein „Thing“, also eine
zählbare Objektinstanz, oder „Stuff“, also eine formlose Region wie Himmel oder
Straße. Damit gelten strengere Anforderungen als bei der
[Instanzsegmentierung](/docs/tasks/instance-segmentation), die Hintergrundpixel
nicht zuweist und überlappende Masken zulässt. Auch gegenüber der
[semantischen Segmentierung](/docs/tasks/semantic-segmentation) ist sie
strenger, denn diese beschriftet zwar jedes Pixel, führt aber sich berührende
Instanzen einer Klasse zusammen.

`panoptic` ist der kanonische Aufgabenschlüssel. Das Suffix `-panoptic` im
Namen einer Checkpoint-Datei wählt diese Aufgabe aus. Beim Laden
veröffentlichter Gewichte ist `task=` daher nicht erforderlich.

`predict()` füllt `result.panoptic`. `.data` ist eine ganzzahlige
Segment-ID-Karte der Form `(H, W)` auf der Canvas des Originalbilds.
`.segments_info` ist eine Liste von Dictionaries mit einem Eintrag pro Segment.
Jeder enthält mindestens `{"id", "category_id"}`. Dabei entspricht `id` einem
Wert in der Karte und `category_id` indiziert `result.names`. `.segment_ids`
führt die vorhandenen IDs in sortierter Reihenfolge auf und
`.segment_mask(id)` gibt die boolesche `(H, W)`-Auswahl für ein Segment zurück.
Die Segment-ID `0` ist der Void-Wert. Diese nicht beschrifteten Pixel werden
aus der Metrik ausgeschlossen und erscheinen nicht in `.segment_ids`.

Die Unterscheidung zwischen Thing und Stuff ist eine Eigenschaft der Kategorie
und nicht des einzelnen Segments. Sie ist in den Kategoriemetadaten des
Labelsatzes gespeichert. Eine Vorhersage-Payload kann sie der Einfachheit
halber als `"isthing"` in jedes Segment kopieren, maßgeblich bleiben jedoch die
Kategoriemetadaten.

## Modelle

[EoMT](/docs/models/eomt) unterstützt diese Aufgabe über `LibreYOLO()`. Die
Familie läuft mit dem Basispaket und stellt auf COCO trainierte panoptische
Checkpoints in den drei Größen s, b und l bereit.

[SenseNova-Vision](/docs/models/sensenova-vision) erzeugt ebenfalls panoptische
Karten. Es handelt sich um ein promptgesteuertes generatives Modell mit eigener
Factory namens `LibreVLM` und eigenem Extra. Wenn kein Vokabular gesetzt ist,
verwendet es die panoptischen COCO-Kategorien, auf die es abgestimmt wurde. Die
Gewichte sind nicht kommerziell nutzbar. Die Latenz pro Bild ist erheblich
höher als bei einem speziell entwickelten Segmentierer, weil jede Vorhersage
eine Diffusionsdecodierung ausführt.

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert.

<code-tabs name="predict" />

`conf` filtert die Query-Auswahl. Unter [Vorhersage](/docs/predict) findest du
Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Datensatzformat

LibreYOLO übernimmt das COCO-Panoptic-Format von Kirillov et al., CVPR 2019,
unverändert. Es gibt kein LibreYOLO-spezifisches panoptisches Layout.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Jedem Bild ist eine RGB-PNG-Datei derselben Auflösung zugeordnet. Die Farbe
jedes Pixels codiert die ID des Segments, zu dem es gehört:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Die Segment-ID `0`, also RGB-Schwarz, steht für Void: nicht beschriftete Pixel,
die eine Vorhersage weder belohnen noch bestrafen. Jedes andere Pixel gehört
genau einem Segment an.

Die JSON führt pro Bild die Segment-ID-PNG und die darin enthaltenen Segmente
auf:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` benennt die PNG-Datei im panoptischen Verzeichnis.
`segments_info[].id` entspricht einem Wert in dieser PNG-Datei. `iscrowd`
kennzeichnet Gruppenregionen. Sie werden nie als False Negatives gezählt, und
eine Vorhersage, die eine solche Region größtenteils abdeckt, gilt nicht als
False Positive. `isthing` befindet sich in `categories` und nie in einem
einzelnen Segment.

Die YAML verweist auf beide:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` und `panoptic_dir` akzeptieren jeweils einen einzelnen Pfad oder
eine Zuordnung pro Split. Rohe COCO-Kategorie-IDs sind normalerweise nicht
zusammenhängend, während Modelle einen zusammenhängenden Bereich `0..nc-1`
vorhersagen. Daher werden IDs anhand des Kategorienamens über `names`
zugeordnet. Fehlt eine JSON-Kategorie in `names`, wird ein Fehler ausgelöst,
statt sie unbemerkt zu verwerfen. Ein solches Verwerfen würde dauerhaft als
False Negative gewertet.

Der kanonische Loader ist `libreyolo.data.PanopticDataset`.

## Training

Derzeit trainiert keine Familie in LibreYOLO panoptische Segmentierung.
`train()` von EoMT löst `NotImplementedError` aus. Panoptische Checkpoints
werden daher wie veröffentlicht verwendet.

## Validierung

`val()` gibt ein einfaches Dictionary mit `metrics/`-Schlüsseln zurück. Die
Werte werden in der Auflösung der Ground Truth über den in der Datensatz-YAML
als `val` bezeichneten Split berechnet. Ein vorhergesagtes und ein tatsächliches
Segment derselben Kategorie stimmen überein, wenn ihre IoU größer als 0.5 ist.
Diese Zuordnung ist eindeutig.

<code-tabs name="val" />

`metrics/PQ` ist die Panoptic Quality und bildet die wichtigste Kennzahl.
Innerhalb einer Kategorie ist sie das Produkt zweier Faktoren. Die Segmentation
Quality ist die mittlere IoU der zugeordneten Segmente und beschreibt, wie gut
die Formen übereinstimmen. Die Recognition Quality ist
`TP / (TP + 0.5 FP + 0.5 FN)`, also der F1-Score der Zuordnung. Sie beschreibt,
wie viele Segmente überhaupt gefunden wurden. Alle drei Werte werden
anschließend über die vorkommenden Kategorien gemittelt und als `metrics/PQ`,
`metrics/SQ` und `metrics/RQ` gemeldet. Die angegebene PQ ist daher der
Mittelwert der Produkte je Kategorie und nicht das Produkt der beiden
angegebenen Mittelwerte.

`metrics/PQ_things` und `metrics/PQ_stuff` mitteln dieselbe PQ pro Kategorie
getrennt über Thing- und Stuff-Kategorien. `metrics/categories` zählt die
vorgekommenen und damit einbezogenen Kategorien. Das Dictionary enthält
außerdem `fitness` als Kopie des PQ-Werts.

## Export

Panoptische Checkpoints können nicht exportiert werden. `export()` löst für
diese Aufgabe `NotImplementedError` aus, weil für die Query-Masken-Ausgabe noch
kein Runtime-Exportvertrag besteht. Die semantische Aufgabe von EoMT kann
exportiert werden. Siehe
[semantische Segmentierung](/docs/tasks/semantic-segmentation) und
[Export und Deployment](/docs/export).

