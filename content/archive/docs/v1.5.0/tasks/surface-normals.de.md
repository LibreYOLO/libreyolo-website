---
title: Oberflächennormalen
seo_title: Schätzung von Oberflächennormalen in LibreYOLO
description: >-
  Sage in LibreYOLO aus einem einzelnen Bild ein dichtes Feld von
  Oberflächennormalen voraus. Verstehe die Kamerarahmen-Konvention, validiere
  den Winkelfehler und exportiere ein Modell.
lead: >-
  Die Schätzung von Oberflächennormalen sagt voraus, in welche Richtung jede
  sichtbare Oberfläche weist. LibreYOLO stellt sie als normal-Aufgabe bereit,
  die ein dichtes Feld von Einheitsvektoren auf der Canvas des Originalbilds
  zurückgibt.
keywords:
  - oberflächennormalen schätzen python
  - normal map aus bild
  - monokulare geometrie
  - winkelfehler metrik
  - dichte normalen vorhersage
last_verified: 1.5.0
snippets:
  predict:
    - label: Normalenfeld vorhersagen
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE, save=True)


        normals = result.normal_map

        print(normals.data.shape)      # (H, W, 3) float32-Einheitsvektoren

        normals.assert_normalized()    # Fehler, wenn ein Pixel keine
        Einheitslänge hat
    - label: Einzelnes Pixel lesen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # OpenCV-Kamerarahmen: +x rechts, +y unten, +z in die Szene. Eine zur
        # Kamera weisende Oberfläche liegt nahe bei (0, 0, -1).
        field = result.normals.data
        h, w = field.shape[:2]
        print(field[h // 2, w // 2])
    - label: Visualisierung speichern
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # plot() rendert das Feld; es ist für Normalen und Kanten definiert.
        result.plot().save("normals.png")
  val:
    - label: Validieren und Metrikschlüssel auslesen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # Grad
        print(metrics["metrics/median_angular_error"])   # Grad
        print(metrics["metrics/within_11_25"])           # Prozent der Pixel
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Exportierte Datei ausführen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet nach Dateiendung. Ein exportiertes Artefakt wird
        # wie ein Checkpoint geladen und gibt dasselbe Results-Objekt zurück.
        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## Definition

Die Aufgabe `normal` sagt aus einem einzelnen RGB-Bild pro Pixel einen
Einheitsvektor mit drei Komponenten voraus: die Richtung, in die die Oberfläche
an diesem Pixel weist. Anders als bei der Tiefe besitzt die Ausgabe keinen
freien Maßstab. Zwei Vorhersagen können daher ohne Ausrichtung direkt
verglichen werden.

Eine Vorhersage füllt `result.normal_map`, eine `NormalMap`-Payload mit einem
float32-Array der Form `(H, W, 3)` auf der Canvas des Originalbilds. Sie ist
auch als `result.normals` erreichbar. Die Vektoren verwenden den
OpenCV-Kamerarahmen von LibreYOLO: `+x` weist nach rechts, `+y` nach unten und
`+z` in die Szene. Sie weisen zur Kamera, sodass eine frontoparallele
Oberfläche als `(0, 0, -1)` dargestellt wird. `.assert_normalized()` prüft, ob
jeder Pixel endlich ist und innerhalb einer Toleranz Einheitslänge besitzt.
`result.boxes` bleibt leer. Daher haben `conf`, `iou` und `max_det` keine
Wirkung. `Results.plot()` deckt diese Aufgabe ab.

## Modelle

Zwei Familien unterstützen `normal`.

[MoGe-2](/docs/models/moge-2) ist die dafür vorgesehene Familie: ein
monokulares Geometriemodell mit einem einzelnen Forward Pass in drei
Encoder-Größen. LibreYOLO kopiert diese Checkpoints nicht in seine eigene
Organisation. Beim Laden wird die passende Größe aus den offiziellen
Repositorys mit festgelegter Revision heruntergeladen und anhand eines
gespeicherten SHA-256-Werts überprüft.

[LibreMODUS](/docs/models/libremodus) erzeugt Normalen als eines der Ziele eines
Any-to-Any-Modells und kann statt eines RGB-Bilds eine Tiefenkarte als Eingabe
verwenden. Dafür sind das Extra `modus` und ein eigenes authentifiziertes Konto
bei Hugging Face erforderlich. Da die Familie weder `val()` noch `export()`
anbietet, nimmt sie nicht an den nachfolgenden Abschnitten zu Validierung und
Export teil.

## Vorhersage

Die Gewichte von MoGe-2 werden bei der ersten Verwendung heruntergeladen und
lokal zwischengespeichert.

<code-tabs name="predict" />

`imgsz` muss durch die Patch-Größe des ViT-Encoders teilbar sein. LibreYOLO
prüft dies vor Beginn des Laufs. Bei einer Liste von Bildern wird pro Bild ein
Forward Pass ausgeführt. Für diese Aufgabe gibt es keinen schnellen Pfad mit
gestapelten Batches. Unter [Vorhersage](/docs/predict) findest du Informationen
zu Quellen, Streaming und Ergebnisverarbeitung.

## Datensatzformat

Die Normalenvalidierung ordnet jedem Bild eine dreikanalige 16-Bit-PNG-Datei
mit demselben Stammnamen und derselben Auflösung sowie optional eine
Gültigkeitsmaske zu.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

Die Ziel-PNG ist exakt dreikanaliges `uint16`, dessen Kanäle als RGB
gespeichert sind. Die Decodierung lautet `n = png / 65535 * 2 - 1`, gefolgt
von der erneuten Normalisierung jedes Vektors. Die decodierten Vektoren
verwenden denselben OpenCV-Kamerarahmen wie die Vorhersagen. Ein Maskenpixel
gilt als gültig, wenn es ungleich null ist. Ohne Maskendatei ist jeder endliche,
nicht nullwertige decodierte Vektor gültig. Ungültige und aufgefüllte
Zielpixel werden intern als `(0, 0, 0)` gehalten und tragen nie zu einer Metrik
bei. Unter [Datensatzformate](/docs/reference/dataset-formats) findest du den
vollständigen Vertrag.

## Training

Keine der beiden Normalenfamilien implementiert Training. `train()` löst bei
beiden `NotImplementedError` aus. Die Seite zu MoGe-2 verweist für Vorhersage,
Validierung und Export auf die festgelegten offiziellen Checkpoints.

## Validierung

`val()` misst den Winkel zwischen jedem vorhergesagten Vektor und dem
zugehörigen Ground-Truth-Vektor über die vom Datensatz als gültig markierten
Pixel.

<code-tabs name="val" />

`metrics/mean_angular_error` und `metrics/median_angular_error` geben diesen
Winkel in Grad an. Niedrigere Werte sind besser. `metrics/within_11_25`,
`metrics/within_22_5` und `metrics/within_30` geben den Prozentanteil gültiger
Pixel an, deren Winkelfehler innerhalb von 11.25, 22.5 beziehungsweise 30 Grad
liegt. Höhere Werte sind besser. Beachte die Einheit: Diese drei Werte sind
Prozentangaben und keine Bruchteile. `fitness` ist
`metrics/within_11_25` geteilt durch 100. Damit erfolgt die Auswahl des besten
Checkpoints auf derselben Skala `[0, 1]` wie bei allen anderen Aufgaben.

## Export

Ein exportiertes Normalenmodell wird anhand seiner Dateiendung wieder über
`LibreYOLO()` geladen. Eine `.onnx`-Datei verhält sich daher wie ein Checkpoint
und gibt dieselben `Results` zurück.

<code-tabs name="export" />

Der Normalenexport verwendet einen Runtime-Vertrag mit fester Auflösung und
Batch-Größe 1. `dynamic` und ein anderer Wert als 1 für `batch` werden
abgelehnt. `imgsz` muss durch die Patch-Größe des Encoders teilbar sein. Die
Unterstützung der einzelnen Formate wird auf der
[MoGe-2-Seite](/docs/models/moge-2) und in der
[vollständigen Exportmatrix](/docs/reference/export-matrix) beschrieben.
[Export](/docs/export) führt die von allen Formaten akzeptierten Argumente auf.

