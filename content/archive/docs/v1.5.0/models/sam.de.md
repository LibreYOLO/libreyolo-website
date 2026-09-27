---
title: SAM
families:
  - sam
seo_title: 'SAM (Segment Anything): Masken in LibreYOLO vorhersagen'
description: >-
  Nutze SAM in LibreYOLO für Prompt-basierte Punkt- und Boxsegmentierung.
  Installiere die Checkpoints Base, Large und Huge unter Apache-2.0 und sage
  damit vorher.
lead: >-
  SAM (Segment Anything) wandelt einen Punkt- oder Boxklick in eine Objektmaske
  um. LibreYOLO lädt es über eine eigene LibreSAM-Factory, getrennt von der
  Detektor-Factory LibreYOLO(), weil ein Prompt-basiertes Modell eine andere
  Aufrufform benötigt.
keywords:
  - sam
  - segment anything
  - prompt-basierte segmentierung
  - interaktive segmentierung
  - punkt prompt
  - box prompt
  - meta ai
last_verified: 1.5.0
snippets:
  predict:
    - label: Punkt- und Box-Prompts
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "base" lädt facebook/sam-vit-base bei der ersten Verwendung
        automatisch.

        # Weitere Größen: "large", "huge" (auch "b"/"l"/"h").

        model = LibreSAM("base")


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
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Der Bild-Encoder ist der aufwendige Teil. set_image() führt ihn einmal
        aus;

        # jeder folgende predict()-Aufruf nutzt das zwischengespeicherte
        Embedding.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f8904d241ef8a929
---

## Installation

SAM benötigt das Zusatzpaket `sam`, das `transformers` und `timm` installiert.

```bash
pip install "libreyolo[sam]"
```

## Vorhersage

`LibreSAM(...)` ist ein separater Einstiegspunkt neben `LibreYOLO(...)`. Er
gibt einen Prompt-basierten Segmentierer statt eines Detektors zurück, weil ein
Vorwärtslauf ohne räumlichen Prompt hier keine Aussagekraft hat. Für diese
Familie gibt es keinen CLI-Befehl `libreyolo predict`. Verwende die Python-API.

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
diese Familie alle `NotImplementedError` aus. SAM unterstützt in LibreYOLO nur
Vorhersagen, und Video-Tracking liegt außerhalb des Umfangs. Unter
[Vorhersage](/docs/predict) findest du die Quelltypen.

## Varianten

Es gibt drei ViT-Bild-Encoder-Größen: Base, Large und Huge, alle mit einer
festen Eingabe von 1024 px. Für diese Familie wurde noch kein Accuracy- oder
Latenzbenchmark veröffentlicht. Die Wahl der Größe tauscht daher direkt
Encoder-Gewicht gegen Maskenqualität. Base lässt sich am schnellsten kodieren,
Huge ist am schwersten.

## Lizenzierung

<provenance-box>

LibreYOLO hostet keine eigene Kopie der SAM-1-Gewichte. `LibreSAM("base")`,
`"large"` und `"huge"` laden direkt aus Metas eigenen Repositorys
`facebook/sam-vit-base`, `facebook/sam-vit-large` und `facebook/sam-vit-huge`
auf Hugging Face. Dort ist jedes unabhängig von LibreYOLO als Apache-2.0
gekennzeichnet.

</provenance-box>

## Zitieren

<citation-block />
