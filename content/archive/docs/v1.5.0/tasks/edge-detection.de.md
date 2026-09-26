---
title: Kantenerkennung
seo_title: Kantenerkennung in LibreYOLO
description: >-
  Sage in LibreYOLO aus einem Bild eine dichte Kantenwahrscheinlichkeitskarte
  vorher. Konvertiere einen Checkpoint, schwelle die Karte, validiere mit ODS
  und OIS und exportiere.
lead: >-
  Die Kantenerkennung sagt vorher, wie wahrscheinlich jedes Pixel auf einer
  Objektgrenze liegt. LibreYOLO stellt sie als edge-Task bereit, der eine dichte
  Wahrscheinlichkeitskarte auf der ursprünglichen Bildfläche zurückgibt statt
  einer Menge von Liniensegmenten.
keywords:
  - kantenerkennung python
  - boundary detection deep learning
  - kantenwahrscheinlichkeitskarte
  - ODS OIS F-measure
  - dichte kantenvorhersage
last_verified: 1.5.0
snippets:
  predict:
    - label: Eine Kantenkarte vorhersagen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # LibreYOLO liefert keinen Edge-Checkpoint; erst konvertieren (unten).
        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)          # (H, W) float32 in [0, 1]
        print(edges.binary(0.5).sum())    # Anzahl Kantenpixel bei 0.5
    - label: Eigenen Schwellenwert wählen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # Die Karte bleibt kontinuierlich, der Schwellenwert ist deine Wahl.
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Die Visualisierung speichern
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot() zeichnet die Karte; für Edge- und Normal-Ergebnisse definiert.
        result.plot().save("edges.png")
  val:
    - label: Validieren und die Metrik-Keys lesen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # fitness
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Sweep und Match-Toleranz ändern
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Die exportierte Datei ausführen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung: Ein exportiertes Artefakt
        # lädt wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Definition

Der `edge`-Task sagt aus einem einzelnen RGB-Bild eine Wahrscheinlichkeit pro
Pixel vorher: `0` bedeutet keine Kante und `1` bedeutet Kante. Die Karte bleibt
kontinuierlich, deshalb bleibt die Wahl des Schwellenwerts, der sie in ein
binäres Grenzbild verwandelt, beim Aufrufer, und der richtige Schwellenwert
hängt vom Datensatz und von der nachgelagerten Verwendung ab.

Eine Vorhersage füllt `result.edges`, ein `EdgeMap`-Payload mit einem `(H, W)`
float32-Array in `[0, 1]` auf der ursprünglichen Bildfläche. `.array` liefert
diese Karte als NumPy und `.binary(threshold)` eine boolesche Maske.
`result.boxes` bleibt leer, deshalb haben `conf`, `iou` und `max_det` keine
Wirkung. `Results.plot()` deckt diese Aufgabe ab und zeichnet die Karte direkt.

## Modelle

Drei Familien bedienen `edge`.

[DexiNed](/docs/models/dexined), das Dense Extreme Inception Network, verschmilzt
mehrere Seitenausgaben zu einer Wahrscheinlichkeitskarte und läuft nativ bei
352 px.

[TEED](/docs/models/teed), der Tiny and Efficient Edge Detector, ist ein kleines
Netz bei denselben nativen 352 px, mit einem Downsample-Stride von 4 gegenüber
16 bei DexiNed, deshalb akzeptiert es mehr Werte für `imgsz`.

[LibreMODUS](/docs/models/libremodus) erzeugt Kanten im Canny-Stil als eines der
Ziele eines Any-to-Any-Modells. Es braucht das Extra `modus` und deinen eigenen
authentifizierten Hugging-Face-Account, und es bietet weder `val()` noch
`export()`, nimmt also an den Abschnitten zu Validierung und Export unten nicht
teil.

## Vorhersage

LibreYOLO veröffentlicht keinen Edge-Checkpoint. Die offiziell veröffentlichten
Gewichte von DexiNed und TEED sind auf BIPED trainiert, dessen veröffentlichte
Datensatzbedingungen die Nutzung auf nicht kommerzielle Zwecke beschränken,
deshalb spiegelt LibreYOLO sie nicht. Konvertiere einen Checkpoint, für den du
lizenziert bist, und lade die konvertierte Datei dann über ihren Pfad:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

Der Dateiname muss das Task-Suffix `-edge` tragen, damit der Loader ihn erkennt.
`imgsz` muss durch den Downsample-Stride des Netzes teilbar sein, und LibreYOLO
löst einen klaren Fehler aus, der den Teiler nennt, wenn das nicht der Fall ist.
Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und den Umgang mit
Ergebnissen.

## Datensatzformat

Die Kanten-Validierung paart jedes RGB-Bild mit einer einkanaligen Karte
desselben Dateistamms und derselben Auflösung, plus einer optionalen
Gültigkeitsmaske.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Das Ziel ist ein einkanaliges PNG oder TIF, keine RGB-Visualisierung.
Ganzzahlige Karten werden durch das Maximum ihres dtype geteilt; Float-Karten
müssen bereits endlich und in `[0, 1]` sein. Maskenpixel gelten als gültig, wenn
sie ungleich null sind, und gepaddete Pixel tragen nie zu einer Metrik bei.
`edge_invert: true` deckt Quellen ab, die schwarze Kanten auf Weiß speichern.
Siehe [Datensatzformate](/docs/reference/dataset-formats) für den vollständigen
Vertrag.

## Training

Keine Kanten-Familie in LibreYOLO hat eine Trainingsimplementierung: `train()`
löst bei allen dreien `NotImplementedError` aus. Jede Modellseite nennt das
Konvertierungsskript, das einen anderswo trainierten Checkpoint in einen
verwandelt, den LibreYOLO laden kann.

## Validierung

`val()` berichtet die F-Measures im BSDS-Stil. Kontinuierliche Vorhersagen
werden zuerst mit einer Gradienten-Non-Maximum-Suppression in vier Richtungen
ausgedünnt, dann werden vorhergesagte und Ground-Truth-Kantenpixel innerhalb
einer Abstandstoleranz eins zu eins zugeordnet.

<code-tabs name="val" />

`metrics/ODS` ist das F-Measure im Optimal Dataset Scale: Die Trefferzahlen
werden bei jedem Schwellenwert über den Datensatz gepoolt, und das beste dieser
gepoolten F-Measures wird berichtet. Es ist zugleich `fitness`, die Zahl, die
die Auswahl des besten Checkpoints liest. `metrics/OIS` ist das F-Measure im
Optimal Image Scale, der Mittelwert über die Bilder aus dem jeweils besten
F-Measure jedes Bildes, jedes Bild darf sich also seinen eigenen Schwellenwert
suchen. `metrics/best_threshold` ist der eine Schwellenwert, der ODS erzeugt
hat, und genau den solltest du bei der Inferenz in `edges.binary()`
wiederverwenden.

Zwei Argumente formen den Sweep. `edge_thresholds` ist die Menge der geprüften
Schwellenwerte, standardmäßig 0.01 bis 0.99 in Hundertstelschritten.
`edge_max_dist` ist die Match-Toleranz als Anteil der Bilddiagonale,
standardmäßig `0.0075`; ein Paar, das weiter auseinanderliegt, ist kein Treffer.

## Export

Ein exportiertes Kantenmodell lädt über `LibreYOLO()` anhand seiner Dateiendung
zurück, deshalb verhält sich eine `.onnx`-Datei wie ein Checkpoint und liefert
dieselben `Results`.

<code-tabs name="export" />

Der Kanten-Export nutzt einen Runtime-Vertrag mit fester Auflösung und Batch 1:
`dynamic` und ein `batch` ungleich 1 werden zurückgewiesen, und der exportierte
Graph gibt eine einzelne fusionierte Wahrscheinlichkeitskarte aus. Die Abdeckung
pro Format steht auf den Seiten [DexiNed](/docs/models/dexined) und
[TEED](/docs/models/teed) sowie in der
[vollständigen Export-Matrix](/docs/reference/export-matrix).
[Export](/docs/export) listet die Argumente auf, die jedes Format akzeptiert.
