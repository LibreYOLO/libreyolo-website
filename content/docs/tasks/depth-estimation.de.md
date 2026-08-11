---
title: Tiefenschätzung
seo_title: Monokulare Tiefenschätzung in LibreYOLO
description: >-
  Sage in LibreYOLO aus einem Bild eine dichte relative Tiefenkarte vorher.
  Vergleiche die Tiefen-Familien, lies die Tiefenmetriken und exportiere ein
  Tiefenmodell.
lead: >-
  Die Tiefenschätzung sagt anhand eines einzelnen Bildes vorher, wie weit jedes
  Pixel von der Kamera entfernt ist. LibreYOLO stellt sie als depth-Task bereit,
  der eine dichte relative inverse Tiefenkarte auf der ursprünglichen Bildfläche
  zurückgibt.
keywords:
  - monokulare tiefenschätzung python
  - tiefenkarte aus einem bild
  - relative tiefe modell
  - depth anything libreyolo
  - dichte tiefenvorhersage
last_verified: 1.5.0
snippets:
  predict:
    - label: Eine Tiefenkarte vorhersagen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) auf der Originalfläche
        print(depth.min, depth.max, depth.mean)
    - label: Mit den Werten arbeiten
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE)

        depth = result.depth_map
        raw = depth.data          # näher = größer; keine Einheit, keine Skala
        gray = depth.normalized() # auf [0, 1] skaliert, für die Anzeige
        print(raw.shape, float(gray.max()))
    - label: Eine kompakte Alternative
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Gleicher Task-Vertrag, ein viel kleineres Netz für Edge-Runtimes.
        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
  val:
    - label: Validieren und die Metrik-Keys lesen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # fitness
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Die exportierte Datei ausführen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung: Ein exportiertes Artefakt
        # lädt wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## Definition

Der `depth`-Task sagt aus einem einzelnen RGB-Bild einen Wert pro Pixel vorher.
LibreYOLO definiert diesen Wert als relative inverse Tiefe: größer bedeutet
näher an der Kamera, und die Zahlen tragen weder eine metrische Einheit noch
eine Skala, die über zwei Bilder hinweg gilt. Die Tiefe zweier Pixel derselben
Vorhersage zu vergleichen ist sinnvoll; einen Wert mit einem Wert aus einem
anderen Bild zu vergleichen ist es nicht.

Eine Vorhersage füllt `result.depth_map`, ein `DepthMap`-Payload mit einem
`(H, W)`-Array auf der ursprünglichen Bildfläche. `.min`, `.max` und `.mean`
lesen die endlichen Werte, und `.normalized()` skaliert die Karte für die
Anzeige auf `[0, 1]`. `result.boxes` bleibt leer, deshalb haben `conf`, `iou`
und `max_det` keine Wirkung, und `save=True` schreibt ein farbcodiertes Bild der
Karte statt eines annotierten Fotos.

## Modelle

Sechs Familien bedienen `depth`.

[Depth Anything V2](/docs/models/depth-anything-v2) kombiniert einen
DINOv2-Encoder mit einem DPT-Decoder und ist hier der Allzweck-Standard. Über
die Größe entscheidet die Lizenzierung genauso wie die Accuracy: Der
Small-Checkpoint steht unter Apache-2.0, während Base und Large nicht
kommerziell sind, prüfe also die Checkpoint-Tabelle auf der Seite, bevor du dich
festlegst.

[Depth Anything 3](/docs/models/depth-anything-3) portiert den Checkpoint
DA3MONO-LARGE, einen schlichten Transformer ohne architektonische
Spezialisierung auf Tiefe.

[ZipDepth](/docs/models/zipdepth) ist die kompakte Stufe: ein
reparametrisierbares CNN, destilliert aus Depth Anything V2 Large, mit einem
zweiten Checkpoint, dessen Decoder Gather- und Unfold-Operationen für
NPU-Compiler vermeidet, denen sie fehlen.

[MiDaS](/docs/models/midas) ist die Arbeitslinie, die das Zero-Shot-Protokoll
für relative Tiefe begründet hat, an dem die anderen Familien gemessen werden.
Es ist die eine Tiefen-Familie, die LibreYOLO nicht erneut veröffentlicht: Wer
einen Checkpoint anfordert, lädt das offizielle Asset aus dem GitHub-Release der
Autoren und prüft einen fest hinterlegten SHA-256.

[LibreMODUS](/docs/models/libremodus) erreicht Tiefe als eines der Ziele eines
Any-to-Any-Modells statt über einen eigenen Head. Es braucht das Extra `modus`
und deinen eigenen authentifizierten Hugging-Face-Account, und es bietet weder
`val()` noch `export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) erzeugt die Tiefenkarte über
einen Diffusion-Decode als Bild, aus demselben 7B-Checkpoint, der auch seine
sechs anderen Aufgaben bedient. Es braucht das Extra `sensenova`, und seine
Gewichte sind auf nicht kommerzielle Nutzung beschränkt; die Lizenz steht auf
seiner Seite.

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert, außer bei den beiden oben genannten Familien.

<code-tabs name="predict" />

Die Eingabeauflösung ist je Familie eingeschränkt. Depth Anything V2 und Depth
Anything 3 bauen auf einem DINOv2-Patchraster auf, deshalb muss `imgsz` glatt
durch 14 teilbar sein, was LibreYOLO vor dem Lauf prüft. `Results.plot()` deckt
diese Aufgabe nicht ab; die Methode ist nur für Oberflächennormalen und Kanten
definiert. Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und den
Umgang mit Ergebnissen.

## Datensatzformat

Die Tiefen-Validierung paart jedes Bild mit einer dichten einkanaligen
Tiefenkarte derselben Auflösung, die gefunden wird, indem das Tiefenverzeichnis
in den Bildpfad eingesetzt wird.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Karten sind einkanaliges PNG oder TIF oder `.npy`. Die Werte sind schlichte
Tiefe in einer Einheit, die der Datensatz konsistent hält, und Pixel mit `0`,
negativen, NaN- oder unendlichen Werten markieren ungültige Samples, die aus den
Metriken ausgeschlossen werden. Ganzzahlige Karten werden durch `depth_scale`
geteilt, standardmäßig `256.0`, die Konvention für 16-Bit-PNG; Float-Karten aus
`.npy` werden so verwendet, wie sie sind. `depth_stem_suffix` und
`depth_mask_suffix` decken Datensätze ab, die ihre Tiefendateien oder
Gültigkeitsmasken anders benennen. Siehe
[Datensatzformate](/docs/reference/dataset-formats) für den vollständigen
Vertrag.

## Training

Keine Tiefen-Familie in LibreYOLO hat eine Trainingsimplementierung: `train()`
löst bei allen sechs `NotImplementedError` aus. Jede Modellseite nennt das
Konvertierungsskript, das einen upstream trainierten Checkpoint in einen
verwandelt, den LibreYOLO laden kann.

## Validierung

`val()` startet den gemeinsamen Tiefen-Validator. Relative Tiefe hat keine
absolute Skala, deshalb wird jede Vorhersage zuerst mit einer
Kleinste-Quadrate-Anpassung von Skalierung und Verschiebung pro Bild an die
Inverse ihrer Ground Truth angepasst und dann zurück in Tiefe invertiert. Jede
Metrik unten wird pro Bild auf dieser angepassten Karte berechnet und über den
Datensatz gemittelt, wobei nur Pixel zählen, die der Datensatz als gültig
markiert.

<code-tabs name="val" />

`metrics/abs_rel` ist der mittlere absolute relative Fehler, das Residuum
geteilt durch die Ground-Truth-Tiefe, und kleiner ist besser. `metrics/rmse` ist
der quadratische Mittelwertfehler in der Tiefeneinheit des Datensatzes, ebenfalls
kleiner ist besser. `metrics/delta1`, `metrics/delta2` und `metrics/delta3` sind
die Schwellenwert-Accuracies: der Anteil der gültigen Pixel, deren Verhältnis
zur Ground Truth, in der jeweils größeren Richtung genommen, unter 1.25, 1.25
zum Quadrat und 1.25 hoch drei liegt, größer ist also besser. `metrics/delta1`
ist zugleich `fitness`, die Zahl, die die Auswahl des besten Checkpoints liest.

## Export

Ein exportiertes Tiefenmodell lädt über `LibreYOLO()` anhand seiner Dateiendung
zurück, deshalb verhält sich eine `.onnx`- oder `.engine`-Datei wie ein
Checkpoint und liefert dieselben `Results`, mit `depth_map` an der Stelle der
Boxen.

<code-tabs name="export" />

Die Abdeckung unterscheidet sich je Familie, und Depth Anything 3 weist jedes
Format außerhalb seines validierten Satzes zurück, statt eine unvalidierte
Konvertierung zu versuchen. Prüfe die Modellseite und die
[vollständige Export-Matrix](/docs/reference/export-matrix), bevor du dich auf
ein Ziel festlegst. LibreMODUS und SenseNova-Vision exportieren überhaupt nicht.
[Export](/docs/export) listet die Argumente auf, die jedes Format akzeptiert.
