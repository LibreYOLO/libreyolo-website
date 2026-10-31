---
title: Kosmos-2
families:
  - kosmos2
seo_title: 'Kosmos-2 in LibreYOLO: Grounded Object Detection'
description: >-
  Kosmos-2 in LibreYOLO: Installiere das Modell, lege ein offenes Vokabular fest
  und sage mit dem MIT-lizenzierten Modell von Microsoft verankerte Boxen
  voraus.
lead: >-
  Kosmos-2 ist das Grounding-Modell von Microsoft: Es beschreibt ein Bild und
  lokalisiert anschließend jede Nominalphrase dieser Beschreibung mit einer Box.
  LibreYOLO bindet es als Open-Vocabulary-Objektdetektor ein. Gib die
  Klassenliste bei der Vorhersage an.
keywords:
  - Kosmos-2
  - vision-language-modell
  - grounding
  - open-vocabulary-erkennung
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # Jede von der Bibliothek unterstützte Quelle: Datei, Ordner, URL,
        # Webcam-Index, RTSP-Stream oder eine .streams-Liste
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## Installation

Kosmos-2 gehört zur VLM-als-Detektor-Stufe von LibreYOLO. Sie bildet eine
eigene Produktoberfläche mit eigener Factory und unterscheidet sich von den
Checkpoint-basierten Familien. Dafür wird das Extra `vlm` benötigt.

```bash
pip install "libreyolo[vlm]"
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert. LibreYOLO lädt direkt aus dem eigenen Repository
`microsoft/kosmos-2-patch14-224` von Microsoft. Anders als bei Florence-2 ist
hier kein erneuter Upload durch die Community erforderlich.

<code-tabs name="predict" />

Diese Familie wird über die Factory `LibreVLM()` und nicht über `LibreYOLO()`
geladen. VLM-Familien deklarieren keinen Checkpoint-Loader. Daher gilt das auf
anderen Modellseiten beschriebene Routing nach Dateiendung hier nicht.
`set_classes()` legt das Vokabular fest, nach dem Kosmos-2 suchen soll. Die
Einstellung bleibt über alle späteren Aufrufe von `predict()` und `track()`
hinweg aktiv, bis du sie erneut setzt. Kosmos-2 verankert Nominalphrasen,
anstatt ein Label exakt abzugleichen. Der Wrapper von LibreYOLO akzeptiert
deshalb auch Teilübereinstimmungen: Eine Klasse namens `"boat"` passt auch zu
einer generierten Phrase wie „the boats“. Jede Erkennung trägt dieselbe
Platzhalter-Confidence. Die Filterung mit `conf` entscheidet daher nur zwischen
allen oder keinen Ergebnissen, statt diese zu ordnen. `iou` hat hier keine
Wirkung, weil der Wrapper die Erkennungsliste ohne Deduplizierungsschritt direkt
aus den verankerten Entitäten erstellt. `chat()` löst `NotImplementedError`
aus, da Kosmos-2 durch einen `<grounding>`-Prompt und nicht durch ein
Chat-Template gesteuert wird. Die CLI von LibreYOLO deckt diese Stufe nicht ab:
Eine Form wie `libreyolo predict model=...` gibt es dafür nicht. Unter
[Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und
Ergebnisverarbeitung.

## Varianten

Es gibt eine Größe: `kosmos-2-patch14-224` mit 224 px. Sie wird als
`LibreVLM("kosmos-2")` geladen. Das Modell stammt aus dem Jahr 2023, und der
LibreYOLO-Wrapper weist darauf hin, dass sein Grounding gröber als das der
neueren Detektoren dieser Stufe ist.

LibreYOLO trainiert, validiert und exportiert Kosmos-2 nicht: `train()`,
`val()` und `export()` lösen für jede Familie dieser Stufe `NotImplementedError`
aus (siehe die oben genannte Support-Stufe). Führe das Fine-Tuning von Kosmos-2
im Upstream-Projekt durch und lade die resultierenden Gewichte, wenn du ein
fest integriertes benutzerdefiniertes Vokabular benötigst. Prüfe die Ausgabe
von `predict()` visuell statt mit einem Validierungsdurchlauf im COCO-Stil, da
jede Erkennung dieselbe Platzhalter-Confidence trägt.

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />

