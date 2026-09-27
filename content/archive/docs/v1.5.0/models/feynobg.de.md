---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: Hintergrundentfernung in LibreYOLO'
description: >-
  Nutze FeyNobg in LibreYOLO für Hintergrundentfernung und Alpha-Matting, eine
  vertiefte BiRefNet-Variante von Feyn Inc. Installiere, sage vorher und
  validiere.
lead: >-
  Ein Modell zur Hintergrundentfernung von Feyn Inc., das die
  BiRefNet-Architektur vertieft und neu trainiert. LibreYOLO bietet Inferenz und
  Validierung für die Matte-Aufgabe von FeyNobg.
keywords:
  - feynobg
  - hintergrund entfernen
  - dichotomous image segmentation
  - alpha matte
  - image matting
  - bild freistellen
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Freistellung
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: Quell-RGB plus Matte als Alphakanal.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreFeyNobgl-matte.pt")


        # Statt einer Datensatz-YAML funktioniert auch ein Verzeichnis mit
        images/

        # und einem automatisch erkannten Matte-Verzeichnis (mattes/, matte/,
        gt/,

        # masks/, mask/ oder alpha/).

        metrics = model.val(data="my-matte-dataset/")


        print(metrics["metrics/MAE"])

        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Installation

FeyNobg benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Der Checkpoint wird bei der ersten Verwendung von der LibreYOLO-Organisation
auf Hugging Face heruntergeladen und wie bei allen anderen Familien lokal
zwischengespeichert. Er ist jedoch noch nicht in der Checkpoint-Tabelle auf
dieser Seite aufgeführt.

<code-tabs name="predict" />

Ein Matte-Ergebnis enthält keine Boxen. `result.matte` ist ein dichtes
float32-Array der Form `(H, W)` im Bereich `[0, 1]`, wobei 1 vollständig zum
Vordergrund und 0 vollständig zum Hintergrund gehört. Anders als eine binäre
Maske bewahrt die weiche Matte geglättete Kantendetails wie Haare und Fell.
`result.cutout()` kombiniert das Quellbild mit diesem Alphakanal zu einem
RGBA-Array. `result.save(path)` (oder `save=True` beim Vorhersageaufruf)
schreibt es direkt als PNG mit transparentem Hintergrund. Das Modell läuft
auf einer festen nativen Arbeitsfläche von 1024x1024. Eine andere Auflösung
wird nicht unterstützt, weil die Tabellen der relativen Positionen im
Swin-Backbone daran gebunden sind. Bei einer Abweichung werden sie schlecht
interpoliert, statt einen Fehler auszulösen. Unter [Vorhersage](/docs/predict)
findest du Quellen, Streaming und die Verarbeitung von Ergebnissen.

## Varianten

Es gibt eine veröffentlichte Größe `l` mit einem Backbone aus dem Swin-L-Tier.
FeyNobg übernimmt die BiRefNet-Architektur, vertieft deren dritte Swin-Stufe
vor dem erneuten Training von 18 auf 24 Blöcke und behält den restlichen Ablauf
bei. Die LibreYOLO-Portierung nutzt daher den Vorwärtspfad, die Vorverarbeitung
und den Ausgabevertrag mit einem einzelnen Logit von BiRefNet erneut.
Vorhersage, Validierung und Checkpoint-Verarbeitung verhalten sich wie bei der
Familie `birefnet`.

## Validierung

`val()` meldet zwei Metriken auf einem gepaarten Bild-und-Matte-Ordner. Beide
liegen im Bereich `[0, 1]` und sind auflösungsunabhängig: MAE ist der mittlere
absolute Fehler gegenüber dem Ground-Truth-Alpha (niedriger ist besser).
S-Measure (Fan et al., ICCV 2017) misst strukturelle Ähnlichkeit und würdigt die
Erhaltung der Form und Öffnungen eines Motivs, die der pixelweise MAE allein
nicht erfasst (höher ist besser). Die Validierung ruft die eigene
`predict`-Methode des Modells auf und verwendet dadurch exakt die
Vorverarbeitung der Familie.

<code-tabs name="val" />

Die Validierung ist reine Inferenz. Die Upstream-Bibliothek `nobg` stellt
Trainingscode unter Apache-2.0 bereit. Fine-Tuning erfolgt derzeit dort mit
anschließender Konvertierung des Ergebnisses durch das LibreYOLO-eigene
Konvertierungsskript. Ein Aufruf von `train()` für diese Familie löst einen
Fehler aus, statt einen unvollständigen Trainer zu starten.

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
