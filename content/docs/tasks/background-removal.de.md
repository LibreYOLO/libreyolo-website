---
title: Hintergrundentfernung
seo_title: Hintergrundentfernung in LibreYOLO
description: >-
  Stelle ein Motiv in LibreYOLO von seinem Hintergrund frei. Sage eine weiche
  Alpha-Matte vorher, schreibe ein transparentes PNG und validiere mit MAE und
  S-Measure.
lead: >-
  Die Hintergrundentfernung trennt ein Motiv von allem, was dahinter liegt.
  LibreYOLO stellt sie als matte-Task bereit, der pro Pixel einen weichen
  Alphawert zurückgibt statt einer harten Vordergrundmaske.
keywords:
  - hintergrund entfernen python
  - alpha matting modell
  - dichotomous image segmentation
  - transparentes png freisteller
  - weiche alpha matte
last_verified: "1.6.0"
snippets:
  predict:
    - label: Eine Matte vorhersagen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 in [0, 1]
    - label: Ein transparentes PNG schreiben
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() legt die Matte als Alphakanal über das Quellbild.
        result.save("subject.png")

        rgba = result.cutout()   # dasselbe (H, W, 4) uint8-Array im Speicher
        print(rgba.shape)
    - label: Auf einen neuen Hintergrund komponieren
      language: python
      code: >
        import numpy as np

        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        result = model(SAMPLE_IMAGE)


        rgba = result.cutout()

        alpha = rgba[..., 3:4].astype(np.float32) / 255.0

        backdrop = np.full_like(rgba[..., :3], 255)          # weiß

        composited = (rgba[..., :3] * alpha + backdrop * (1 -
        alpha)).astype(np.uint8)

        print(composited.shape)
  val:
    - label: Validieren und die Metrik-Keys lesen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Statt einer Datensatz-YAML genügt ein Verzeichnis mit images/ und
        # einem Matte-Verzeichnis.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # kleiner ist besser
        print(metrics["metrics/Smeasure"])   # fitness, größer ist besser
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Die exportierte Datei ausführen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung: Ein exportiertes Artefakt
        # lädt wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: "69fbc1d967b2d546"
---

## Definition

Der `matte`-Task sagt aus einem einzelnen RGB-Bild einen Alphawert pro Pixel
vorher: `1` ist vollständig Vordergrund und `0` vollständig Hintergrund. Der Wert
ist kontinuierlich statt binär, und genau darum geht es bei dieser Aufgabe. Eine
harte Maske ist nur einen Schwellenwert entfernt, bei 0.5, während die weiche
Matte zusätzlich die teilweise Deckung an Haaren, Fell und bewegungsunscharfen
Kanten trägt, die eine binäre Maske wegwirft.

Eine Vorhersage füllt `result.matte`, ein `Matte`-Payload mit einem `(H, W)`
float32-Array in `[0, 1]` auf der ursprünglichen Bildfläche, über `.array` als
NumPy erreichbar. `result.cutout()` kombiniert das Quellbild mit diesem Alpha zu
einem `(H, W, 4)` uint8-RGBA-Array, und `result.save(path)` schreibt dasselbe in
ein PNG mit transparentem Hintergrund. `result.boxes` bleibt leer, deshalb haben
`conf`, `iou` und `max_det` keine Wirkung.

## Modelle

BiRefNet und FeyNobg verwenden denselben Forward-Pfad.

[BiRefNet](/docs/models/birefnet) ist das bilaterale Referenznetz, um das die
Aufgabe herum gebaut ist, hier als ein Checkpoint der Swin-L-Stufe
veröffentlicht.

[FeyNobg](/docs/models/feynobg) ist die vertiefte Variante von Feyn Inc.: die
Architektur von BiRefNet, deren dritte Swin-Stufe von 18 auf 24 Blöcke
gewachsen ist und die danach neu trainiert wurde. LibreYOLO nutzt dafür den
Forward-Pfad, die Vorverarbeitung und die Single-Logit-Ausgabe von BiRefNet
weiter, deshalb verhalten sich Vorhersage, Validierung und Checkpoint-Handling
identisch; die Gewichte und die Familienidentität gehören FeyNobg selbst.

Die beiden tragen unterschiedliche Lizenzen für ihre Gewichte. Beide sind auf
den Modellseiten angegeben, und maßgeblich ist die Lizenz im
Hugging-Face-Repository des konkreten Checkpoints.

[BEN2](/docs/models/ben2) ergänzt Hintergrundentfernung bei fester Auflösung 1024. [ViTMatte](/docs/models/vitmatte) nimmt ein Bild und eine dreistufige `trimap=`, die Hintergrund, unbekannte Bereiche und Vordergrundpixel kennzeichnet.

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

BiRefNet und FeyNobg arbeiten mit einer festen nativen Bildfläche von 1024x1024 und skalieren die Matte auf das Originalbild zurück. Andere Auflösungen werden nicht unterstützt: Die relativen Positionstabellen des Swin-Backbones sind an diese Größe gebunden; eine Abweichung führt zu schlechter Interpolation statt zu einem Fehler. `Results.save()` verwendet das Quellbild für freigestellte Bilder und lädt es aus `Results.path` neu, sofern du keines übergibst. Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und Ergebnisverarbeitung.

`Results.save()` speichert freigestellte Bilder als RGBA. `plot()` stellt ein Bild zur Prüfung dar. BEN2 unterstützt native Batch-Vorhersagen; ViTMatte benötigt eine Führung für ein einzelnes Bild.

## Datensatzformat

Die Matte-Validierung paart jedes RGB-Bild mit einer einkanaligen
Ground-Truth-Alpha-Matte mit demselben Dateistamm, wobei 0 Hintergrund und 255
Vordergrund ist.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Es genügt, dieses Wurzelverzeichnis als `data=` zu übergeben: Das
Matte-Verzeichnis wird unter `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` und
`alpha/` automatisch erkannt. Eine Datensatz-YAML ist die Alternative, mit `path`
sowie `val_images` und `val_mattes`, die Verzeichnisse relativ dazu benennen:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` und `names` sind Schema-Platzhalter; ein Matte-Modell liefert
`Results.matte`, keine Detektionen. Matte-Werte werden durch Division durch 255
als Alpha in `[0, 1]` gelesen, und eine Matte, deren Form von der
Vorhersagefläche abweicht, wird bilinear passend skaliert. Siehe
[Datensatzformate](/docs/reference/dataset-formats) für den vollständigen
Vertrag.

## Training

Diese Matting-Familien haben keine Trainingsimplementierung. Die Exportunterstützung unterscheidet sich je nach Familie; ViTMatte unterstützt keinen Export. Jede Modellseite nennt das Upstream-Projekt mit Trainingscode und das Konvertierungsskript zum Zurückbringen eines Checkpoints.

## Validierung

`val()` treibt das eigene `predict` des Modells an, deshalb nutzt die
Validierung genau die Vorverarbeitung der Familie, und beide Metriken werden auf
der ursprünglichen Bildfläche berechnet.

<code-tabs name="val" />

`metrics/MAE` ist der mittlere absolute Fehler gegenüber dem
Ground-Truth-Alpha, in `[0, 1]`, und kleiner ist besser. `metrics/Smeasure` ist
das S-Measure von Fan et al. (ICCV 2017), eine strukturelle Ähnlichkeit, die es
honoriert, die Form des Motivs und seine Löcher richtig zu treffen, was ein
reiner Mittelwert pro Pixel übersieht; größer ist besser. Das S-Measure ist
zugleich `fitness`, die Zahl, die die Auswahl des besten Checkpoints liest.
Keine der beiden Metriken hängt von der Auflösung ab.

## Export

Ein exportiertes Matte-Modell lädt über `LibreYOLO()` anhand seiner Dateiendung
zurück, deshalb verhält sich das Artefakt wie ein Checkpoint und liefert
dieselben `Results`.

<code-tabs name="export" />

TorchScript ist der validierte Pfad für diese Aufgabe. Die ONNX-Konvertierung
läuft durch, hat dieselbe Paritätshürde aber nicht genommen, und die übrigen
Formate sind nicht verfügbar. Die Abdeckung pro Format steht auf den Seiten
[BiRefNet](/docs/models/birefnet) und [FeyNobg](/docs/models/feynobg) sowie in
der [vollständigen Export-Matrix](/docs/reference/export-matrix).
