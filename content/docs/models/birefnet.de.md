---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: Hintergrundentfernung und Matting in LibreYOLO'
description: >-
  Nutze BiRefNet in LibreYOLO für Hintergrundentfernung und dichotome
  Bildsegmentierung. Installiere, sage vorher, validiere und exportiere den
  general-Checkpoint.
lead: >-
  Ein bilaterales Referenznetz, das eine weiche Alpha-Matte vorhersagt und damit
  ein Motiv von seinem Hintergrund trennt. LibreYOLO liefert Inferenz und
  Validierung für den matte-Task von BiRefNet.
keywords:
  - BiRefNet
  - hintergrund entfernen python
  - background removal python
  - dichotomous image segmentation
  - alpha matte
  - image matting
  - bild freistellen transparenter hintergrund
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Freisteller
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: Quell-RGB plus die Matte als Alphakanal.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Statt einer Datensatz-YAML funktioniert auch ein Verzeichnis mit
        # images/ und einem automatisch erkannten Matte-Verzeichnis
        # (mattes/, matte/, gt/, masks/, mask/ oder alpha/).
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung, ein exportiertes Artefakt
        # lädt also wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Installation

BiRefNet braucht kein optionales Extra. Alles, was es importiert, steckt in
der Basisinstallation.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

Ein Matte-Ergebnis trägt keine Boxen; `result.matte` ist ein dichtes
float32-Array der Form `(H, W)` in `[0, 1]`, wobei 1 vollständig Vordergrund
und 0 vollständig Hintergrund bedeutet. Anders als eine binäre Maske behält
die weiche Matte kantengeglättete Details wie Haare und Fell. `result.cutout()`
setzt das Quellbild mit diesem Alphakanal zu einem RGBA-Array zusammen, und
`result.save(path)` (oder `save=True` beim Vorhersageaufruf) schreibt es direkt
in ein PNG mit transparentem Hintergrund. Das Modell arbeitet mit einer festen
nativen Canvas-Größe von 1024x1024; eine andere Auflösung wird nicht
unterstützt, weil die Tabellen für relative Positionen des Swin-Backbones daran
gebunden sind und eine Abweichung sie schlecht interpoliert, statt einen Fehler
auszulösen. Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und den
Umgang mit Ergebnissen.

## Varianten

Ein veröffentlichter Checkpoint, `l`, das BiRefNet-general-Modell der
Swin-L-Stufe und upstream die Standardwahl für Qualität. Der Code der Familie
unterstützt außerdem eine Swin-T-Lite-Stufe, `t`, aber dafür ist noch keine
LibreYOLO-Konvertierung veröffentlicht.

## Validierung

`val()` meldet zwei Metriken über einen gepaarten Ordner aus Bildern und
Matten, beide in `[0, 1]` und unabhängig von der Auflösung: MAE, den mittleren
absoluten Fehler gegenüber dem Alpha der Ground Truth (niedriger ist besser),
und S-measure (Fan et al., ICCV 2017), eine strukturelle Ähnlichkeit, die es
honoriert, wenn Form und Löcher des Motivs erhalten bleiben, was der reine
Pixel-MAE übersieht (höher ist besser). Die Validierung läuft über das
`predict` des Modells selbst und nutzt damit exakt das Preprocessing der
Familie.

<code-tabs name="val" />

Die Validierung ist reine Inferenz; Fine-Tuning ist ein dokumentierter
nächster Schritt und keine ausgelieferte Funktion (siehe Vorhersage für die
genaue Auflösungsbeschränkung, die jeder künftige Trainer erben würde).

## Export

<export-matrix />

Ein exportiertes Artefakt lädt über `LibreYOLO()` anhand seiner Dateiendung
wieder, eine `.onnx`-Datei verhält sich also wie ein Checkpoint und liefert
dasselbe `Results`. TorchScript ist der validierte Weg; die
ONNX-Konvertierung läuft, hat aber dieselbe Paritätshürde nicht genommen.
[Export](/docs/export) listet die Argumente auf, die jedes Format akzeptiert,
und die Extras, die einige davon hinzufügen.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
