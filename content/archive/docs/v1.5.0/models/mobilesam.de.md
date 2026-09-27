---
title: MobileSAM
families:
  - mobilesam
seo_title: 'MobileSAM: leichte Prompt-basierte Segmentierung in LibreYOLO'
description: >-
  Nutze MobileSAM in LibreYOLO für Prompt-basierte Punkt- und Boxsegmentierung
  mit einem TinyViT-Encoder. Installiere den Tiny-Checkpoint unter Apache-2.0
  und sage damit vorher.
lead: >-
  MobileSAM ersetzt den ViT-H-Bild-Encoder von SAM durch einen destillierten
  TinyViT-Encoder. Dadurch läuft derselbe Prompt-basierte Ablauf mit Punkten und
  Boxen auf leichterer Hardware. LibreYOLO bietet eine native Portierung über
  eine eigene LibreSAM-Factory, getrennt von der Detektor-Factory LibreYOLO().
keywords:
  - mobilesam
  - segment anything leicht
  - tinyvit
  - prompt-basierte segmentierung
  - interaktive segmentierung
  - punkt prompt
  - box prompt
  - leichte segmentierung
last_verified: 1.5.0
snippets:
  predict:
    - label: Punkt- und Box-Prompts
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # MobileSAM hat nur die Größe "tiny", daher ist kein weiterer Alias
        nötig.

        model = LibreSAM("mobilesam")


        # Ein Punkt-Prompt: [x, y] in Pixelkoordinaten, Label 1 = Vordergrund.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # Polygon pro Maske

        print(result.boxes.xyxy)    # aus der Maske abgeleitete enge Box


        # Ein Box-Prompt anstelle eines Punkts.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Ohne Prompt wird das gesamte Bild segmentiert (ein vereinfachter
        automatischer

        # Maskengenerator, nicht die vollständige Referenzimplementierung).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Einmal kodieren, mehrfach prompten'
      language: python
      code: >
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE


        model = LibreMobileSAM()


        # Der Bild-Encoder ist der aufwendige Teil. set_image() führt ihn einmal
        aus;

        # jeder folgende predict()-Aufruf nutzt das zwischengespeicherte
        Embedding.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f96e885d93f72bdd
---

## Installation

MobileSAM benötigt das Zusatzpaket `sam`. Der eigene Gewichtsdownload von
LibreYOLO verwendet weiterhin die Snapshot-Werkzeuge von Hugging Face aus
`transformers`, obwohl die Inferenz auf einem nativen Decoder ohne
`transformers` läuft.

```bash
pip install "libreyolo[sam]"
```

## Vorhersage

`LibreSAM(...)` (oder das familienspezifische `LibreMobileSAM(...)`) ist ein
separater Einstiegspunkt neben `LibreYOLO(...)`. Er gibt einen Prompt-basierten
Segmentierer statt eines Detektors zurück, weil ein Vorwärtslauf ohne räumlichen
Prompt hier keine Aussagekraft hat. Für diese Familie gibt es keinen CLI-Befehl
`libreyolo predict`. Verwende die Python-API.

<code-tabs name="predict" />

Ein Punkt-Prompt akzeptiert `[x, y]` für ein Objekt, `[[x, y], ...]` für mehrere
oder NumPy-Arrays. `labels` kennzeichnet jeden Punkt mit `1` (Vordergrund) oder
`0` (Hintergrund) und verwendet standardmäßig für alle den Vordergrund. Ein
Box-Prompt akzeptiert `[x1, y1, x2, y2]` oder eine Liste von Boxen und erzeugt
eine Maske pro Box. Wenn du beide Prompt-Arten auslässt, wird das gesamte Bild
segmentiert. Dazu wird ein dichtes Raster als Prompt verwendet, und sichere,
nicht überlappende Masken werden beibehalten. Dieser Modus „alles segmentieren“
ist gegenüber dem automatischen Maskengenerator der Referenz vereinfacht und
kann dicht besetzte Szenen zu grob segmentieren. Ein echter Punkt- oder
Box-Prompt ist daher der präzise Pfad. `conf` filtert anhand der vorhergesagten
Maskenqualität (IoU), nicht anhand einer Erkennungs-Confidence. Übergib `0.0`,
um alle Kandidaten zu behalten. `multimask=True` gibt alle drei
Ganzes-gegen-Teil-Mehrdeutigkeitsmasken von SAM pro Prompt zurück statt nur der
besten. `device=` verschiebt das Modell und bei einer aktiven `set_image()`-Sitzung
auch das zwischengespeicherte Embedding. Jede Maske trägt die Klassen-ID `0`
mit dem Namen `"object"`, weil eine Prompt-basierte Maske keine feste
Klassenmenge besitzt. `train()`, `val()`, `export()` und `track()` lösen für
diese Familie alle `NotImplementedError` aus. MobileSAM unterstützt in
LibreYOLO nur Vorhersagen. Unter [Vorhersage](/docs/predict) findest du die
Quelltypen.

## Varianten

Es gibt eine Größe, Tiny, mit einer festen Eingabe von 1024 px. MobileSAM
liefert einen einzelnen TinyViT-Encoder statt der Abstufung Base/Large/Huge von
SAM-1.

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
