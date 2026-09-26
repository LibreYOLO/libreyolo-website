---
title: SAM 2
families:
  - sam2
seo_title: 'SAM 2: Prompt-basierte Bildsegmentierung in LibreYOLO'
description: >-
  Nutze SAM 2 in LibreYOLO für Prompt-basierte Punkt- und Boxsegmentierung.
  Installiere die Checkpoints Tiny, Small, Base-plus und Large unter Apache-2.0
  und sage damit vorher.
lead: >-
  SAM 2 erweitert SAM um eine Streaming-Memory-Architektur für Video und wandelt
  einen Punkt- oder Boxklick in eine Objektmaske um. LibreYOLO unterstützt den
  Bildsegmentierungspfad über eine eigene LibreSAM-Factory, getrennt von der
  Detektor-Factory LibreYOLO().
keywords:
  - sam 2
  - segment anything
  - prompt-basierte segmentierung
  - interaktive segmentierung
  - punkt prompt
  - box prompt
  - meta ai
  - hiera
last_verified: 1.5.0
snippets:
  predict:
    - label: Punkt- und Box-Prompts
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # Größenaliasse: "sam2-tiny", "sam2-small", "sam2-base-plus",

        # "sam2-large" (auch Kurzformen "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l").

        model = LibreSAM("sam2-large")


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
        from libreyolo import LibreSAM2, SAMPLE_IMAGE


        # Die familienspezifische Klasse erhält die Größe ohne Präfix "sam2-".

        model = LibreSAM2("large")


        # Der Bild-Encoder ist der aufwendige Teil. set_image() führt ihn einmal
        aus;

        # jeder folgende predict()-Aufruf nutzt das zwischengespeicherte
        Embedding.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: 2a3090d7ecd533b0
---

## Installation

SAM 2 benötigt das Zusatzpaket `sam`, das `transformers` und `timm` installiert.

```bash
pip install "libreyolo[sam]"
```

## Vorhersage

`LibreSAM(...)` (oder das familienspezifische `LibreSAM2(...)`) ist ein
separater Einstiegspunkt neben `LibreYOLO(...)`. Er gibt einen Prompt-basierten
Segmentierer statt eines Detektors zurück, weil ein Vorwärtslauf ohne räumlichen
Prompt hier keine Aussagekraft hat. Für diese Familie gibt es keinen CLI-Befehl
`libreyolo predict`. Verwende die Python-API. Es wird nur die Bildsegmentierung
unterstützt. Das Video-Memory-Tracking von SAM 2 liegt hier außerhalb des Umfangs.

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

Es gibt vier Größen mit Hiera-Backbone: Tiny, Small, Base-plus und Large, alle
mit derselben Eingabeauflösung. Für diese Familie wurde noch kein Accuracy-
oder Latenzbenchmark veröffentlicht. Die Wahl der Größe tauscht daher direkt
Encoder-Gewicht gegen Maskenqualität. Tiny lässt sich am schnellsten kodieren,
Large ist am schwersten.

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
