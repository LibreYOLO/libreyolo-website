---
title: SAM 3
families:
  - sam3
seo_title: 'SAM 3: Prompt- und Konzeptsegmentierung in LibreYOLO'
description: >-
  Nutze SAM 3 in LibreYOLO für Punkt-, Box- und Textkonzeptsegmentierung.
  Installiere den zugangsbeschränkten Large-Checkpoint unter Metas SAM License
  und sage damit vorher.
lead: >-
  SAM 3 ergänzt die üblichen Punkte und Boxen um einen Textkonzept-Prompt. Eine
  Phrase wie „yellow school bus“ gibt dadurch jede passende Instanz zurück.
  LibreYOLO unterstützt den Bildpfad über eine eigene LibreSAM-Factory, getrennt
  von der Detektor-Factory LibreYOLO().
keywords:
  - sam 3
  - segment anything
  - prompt-basierte segmentierung
  - konzeptsegmentierung
  - text prompt
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


        # "sam3" ist die einzige Größe ("large"). Aliasse: "sam3", "sam-3",
        "sam3-large".

        model = LibreSAM("sam3")


        # Ein Punkt-Prompt: [x, y] in Pixelkoordinaten, Label 1 = Vordergrund.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # Polygon pro Maske

        print(result.boxes.xyxy)    # aus der Maske abgeleitete enge Box


        # Ein Box-Prompt anstelle eines Punkts.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Text-Prompt (Konzept)
      language: python
      code: |
        from libreyolo import LibreSAM3, SAMPLE_IMAGE

        model = LibreSAM3("large")

        # Findet jede zur Phrase passende Instanz, nicht nur ein Objekt.
        # text= schließt points, bboxes, labels und masks gegenseitig aus.
        result = model.predict(SAMPLE_IMAGE, text="a person")
        print(result.names)         # {0: "a person"}
        print(result.boxes.conf)    # PCS-Erkennungs-Score pro Instanz
    - label: 'Einmal kodieren, mehrfach prompten'
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # Der Bild-Encoder ist der aufwendige Teil. set_image() führt ihn einmal
        aus;

        # jeder folgende predict()-Aufruf nutzt das zwischengespeicherte
        Embedding. Ein

        # Aufruf mit text= kodiert intern erneut, weil Tracker und Encoder der

        # Konzeptsegmentierung keinen Cache teilen.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## Installation

SAM 3 benötigt das Zusatzpaket `sam`, das `transformers` und `timm` installiert.

```bash
pip install "libreyolo[sam]"
```

Der Zugriff auf die Gewichte ist beschränkt. Öffne
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), akzeptiere
Metas SAM License und führe vor dem ersten Download `hf auth login` aus (oder
setze `HF_TOKEN`). LibreYOLO protokolliert beim ersten Download dieser Familie
einen Lizenzhinweis.

## Vorhersage

`LibreSAM(...)` (oder das familienspezifische `LibreSAM3(...)`) ist ein
separater Einstiegspunkt neben `LibreYOLO(...)`. Er gibt einen Prompt-basierten
Segmentierer statt eines Detektors zurück, weil ein Vorwärtslauf ohne Prompt
hier keine Aussagekraft hat. Für diese Familie gibt es keinen CLI-Befehl
`libreyolo predict`. Verwende die Python-API. Es wird nur Bildinferenz
unterstützt. Die Videomodelle von SAM 3 liegen hier außerhalb des Umfangs.

<code-tabs name="predict" />

Der Punkt- und Boxpfad entspricht dem Rest der SAM-Familie. Ein Punkt-Prompt
akzeptiert `[x, y]` für ein Objekt oder `[[x, y], ...]` für mehrere. `labels`
kennzeichnet jeden Punkt mit `1` (Vordergrund) oder `0` (Hintergrund). Ein
Box-Prompt akzeptiert `[x1, y1, x2, y2]` oder eine Liste von Boxen. `conf`
filtert auf diesem Pfad anhand der vorhergesagten Maskenqualität (IoU), nicht
anhand einer Erkennungs-Confidence.

Der Pfad `text=` ist die Ergänzung von SAM 3. Ein Konzeptstring gibt über
Promptable Concept Segmentation jede passende Instanz im Bild zurück und kann
nicht mit Punkten, Boxen, Labels oder Masken kombiniert werden. `conf` ist dort
der PCS-Erkennungs-Score statt der Masken-IoU. Der Standardwert verwendet den
eigenen Schwellenwert 0.3 des Modells. `conf=0.0` behält alle Kandidaten bei.
Das zurückgegebene `names` ordnet die Klassen-ID `0` dem angefragten
Konzeptstring zu, weil eine Prompt-basierte Maske sonst keine feste Klassenmenge
besitzt. `device=` verschiebt das Modell und bei einer aktiven
`set_image()`-Sitzung auch das zwischengespeicherte Embedding. `train()`,
`val()`, `export()` und `track()` lösen für diese Familie alle
`NotImplementedError` aus. SAM 3 unterstützt in LibreYOLO nur Vorhersagen, und
Video-Tracking liegt außerhalb des Umfangs. Unter [Vorhersage](/docs/predict)
findest du die Quelltypen.

## Varianten

Es gibt eine Größe, Large, mit einer festen Eingabe von 1008 px. SAM 3.1 wird
nicht unterstützt. Seine Implementierung besitzt eine eigene Lizenz, die nicht
in dieses MIT-Repository übernommen werden kann. Außerdem lädt die von
LibreYOLO verwendete Transformers-Version sein Checkpoint-Format noch nicht.

## Lizenzierung

<provenance-box>

LibreYOLO hostet keine eigene Kopie der SAM-3-Gewichte und verteilt sie nicht.
`LibreSAM("sam3")` lädt sie direkt aus Metas zugangsbeschränktem Repository
`facebook/sam3` auf Hugging Face. Vor dem ersten Download musst du Metas SAM
License akzeptieren und dich authentifizieren.

</provenance-box>

## Zitieren

<citation-block />
