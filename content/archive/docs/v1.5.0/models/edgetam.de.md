---
title: EdgeTAM
families:
  - edgetam
seo_title: 'EdgeTAM: Prompt-basierte Segmentierung auf dem Gerät mit LibreYOLO'
description: >-
  Nutze EdgeTAM in LibreYOLO für schnelle Prompt-basierte Punkt- und
  Boxsegmentierung auf dem Gerät. Installiere den Checkpoint unter Apache-2.0
  und sage damit vorher.
lead: >-
  EdgeTAM ist eine für die Inferenz auf Mobilgeräten beschleunigte Variante von
  SAM 2, die denselben Prompt-basierten Ablauf mit Punkten und Boxen beibehält.
  LibreYOLO unterstützt den Bildsegmentierungspfad über eine eigene
  LibreSAM-Factory, getrennt von der Detektor-Factory LibreYOLO().
keywords:
  - edgetam
  - sam 2 mobilgerät
  - prompt-basierte segmentierung
  - interaktive segmentierung
  - segmentierung auf dem gerät
  - punkt prompt
  - box prompt
  - meta reality labs
last_verified: 1.5.0
snippets:
  predict:
    - label: Punkt- und Box-Prompts
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # EdgeTAM hat eine Größe, "edge". Aliasse: "edgetam", "edge-tam",

        # "edgetam-edge".

        model = LibreSAM("edgetam")


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
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE


        model = LibreEdgeTAM()


        # Der Bild-Encoder ist der aufwendige Teil. set_image() führt ihn einmal
        aus;

        # jeder folgende predict()-Aufruf nutzt das zwischengespeicherte
        Embedding.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: e6cce8faad18e73d
---

## Installation

EdgeTAM benötigt das Zusatzpaket `sam`, das `transformers` und `timm` installiert.

```bash
pip install "libreyolo[sam]"
```

## Vorhersage

`LibreSAM(...)` (oder das familienspezifische `LibreEdgeTAM(...)`) ist ein
separater Einstiegspunkt neben `LibreYOLO(...)`. Er gibt einen Prompt-basierten
Segmentierer statt eines Detektors zurück, weil ein Vorwärtslauf ohne räumlichen
Prompt hier keine Aussagekraft hat. Für diese Familie gibt es keinen CLI-Befehl
`libreyolo predict`. Verwende die Python-API. Es wird nur die Bildsegmentierung
unterstützt. Das Video-Tracking von EdgeTAM liegt hier außerhalb des Umfangs.

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
diese Familie alle `NotImplementedError` aus. LibreYOLO unterstützt hier die
Bildinferenz. Unter [Vorhersage](/docs/predict) findest du die Quelltypen.

## Varianten

Es gibt eine Größe, edge, mit fester Eingabeauflösung. Die Wahl dieser Familie
gegenüber dem restlichen SAM-Tier ist daher eine Hardware- und keine
Größenentscheidung: EdgeTAM wurde speziell für eingeschränkte Inferenz auf dem
Gerät entwickelt.

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
