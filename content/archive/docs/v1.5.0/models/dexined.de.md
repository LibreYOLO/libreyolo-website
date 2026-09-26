---
title: DexiNed
families:
  - dexined
seo_title: 'DexiNed: Kantenerkennung mit eigenem Checkpoint'
description: >-
  Nutze DexiNed in LibreYOLO für die Vorhersage dichter
  Kantenwahrscheinlichkeiten. Konvertiere einen lizenzierten Checkpoint, sage
  anschließend vorher, validiere und exportiere ihn.
lead: >-
  DexiNed (Dense Extreme Inception Network) ist ein faltungsbasiertes Netzwerk,
  das aus einem einzelnen RGB-Bild eine dichte Kantenwahrscheinlichkeitskarte
  vorhersagt. LibreYOLO bindet seine Architektur ausschließlich für die
  Kantenerkennung ein. Die Bibliothek liefert keinen Checkpoint mit.
keywords:
  - dexined kantenerkennung
  - dense extreme inception network
  - edge detection python
  - biped datensatz
  - dichte kantenvorhersage
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)        # (H, W) float32 in [0, 1]

        print(edges.binary(0.5).sum())  # Anzahl der Kantenpixel nach
        Schwellenwert
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreDexiNedb-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # F-Maß mit optimaler Datensatzskala
        print(metrics["metrics/OIS"])   # F-Maß mit optimaler Bildskala
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreDexiNedb-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreDexiNedb-edge.pt format=onnx
        imgsz=352

        libreyolo export model=weights/LibreDexiNedb-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Exportierte Datei verwenden
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: 342597fde3c4ba65
---

## Installation

DexiNed benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

LibreYOLO liefert keinen DexiNed-Checkpoint mit. Die offiziell veröffentlichten
Gewichte wurden auf BIPED trainiert, dessen veröffentlichte Datensatzbedingungen
die Verwendung auf nicht kommerzielle Zwecke beschränken. LibreYOLO spiegelt
sie daher nicht. Konvertiere einen Checkpoint, für dessen Verwendung du eine
Lizenz besitzt, mit `weights/convert_dexined_weights.py`. Das Skript prüft die
Tensor-Schlüssel anhand der Runtime-Architektur, bevor es eine Datei schreibt,
die LibreYOLO direkt laden kann:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` enthält das Ergebnis: ein `(H, W)`-Array vom Typ float32 im
Bereich `[0, 1]`. `.binary(threshold)` gibt eine boolesche Kantenmaske zurück.
Es gibt keine Boxen, daher haben `conf`, `iou` und `max_det` keine Wirkung.
Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

LibreYOLO bietet DexiNed in einer Größe an. Der Benchmark-Testaufbau von
LibreYOLO hat diese Familie noch nicht gemessen, daher gibt es keine
veröffentlichten Vergleichswerte.

## Validierung

`val()` meldet F-Maße für ODS und OIS im BSDS-Stil auf einem gepaarten
Kantendatensatz. Dabei stehen Bilder neben gleichnamigen Kantenkarten. Eine
optionale Gültigkeitsmaske verhindert, dass aufgefüllte Pixel gezählt werden.
`imgsz` muss durch den Downsampling-Stride des Netzwerks teilbar sein. Andernfalls
löst LibreYOLO einen eindeutigen Fehler aus.

<code-tabs name="val" />

## Export

<export-matrix />

Der Kantenexport nutzt einen Runtime-Vertrag mit fester Auflösung und Batch 1:
`dynamic` und ein anderer `batch`-Wert als 1 werden abgelehnt. Der exportierte
Graph gibt eine einzelne zusammengeführte Wahrscheinlichkeitskarte aus. Ein
exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`-Datei verhält sich daher wie ein Checkpoint und gibt
dasselbe `Results`-Objekt zurück.

<code-tabs name="export" />

## Lizenzierung

<provenance-box>

LibreYOLO veröffentlicht keinen DexiNed-Checkpoint. Unter der LibreYOLO-Organisation
wird nichts gespiegelt. Konvertiere stattdessen mit
`weights/convert_dexined_weights.py` einen Checkpoint, für den du eine Lizenz
besitzt.

</provenance-box>

## Zitieren

<citation-block />
