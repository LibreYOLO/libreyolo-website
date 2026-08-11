---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 in LibreYOLO: Open-Vocabulary-Erkennung'
description: >-
  Florence-2 in LibreYOLO: Installiere Microsofts MIT-lizenziertes
  Vision-Modell, lege ein offenes Vokabular fest und sage Boxen vorher.
lead: >-
  Florence-2 ist Microsofts Vision-Foundation-Modell. Statt einen festen
  Erkennungs-Head zu durchlaufen, wird es mit einem Aufgaben-Token gepromptet.
  LibreYOLO bindet es als Open-Vocabulary-Objektdetektor ein: Übergib die
  Klassenliste bei der Vorhersage.
keywords:
  - florence-2 python
  - vision-language model
  - open-vocabulary-erkennung
  - grounding
  - microsoft florence
  - vlm objekterkennung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("florence-2-base")

        model.set_classes(["car", "person", "traffic light"])


        # Jede von der Bibliothek unterstützte Quelle: Datei, Ordner, URL,
        Webcam-Index,

        # RTSP-Stream oder eine .streams-Liste

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Installation

Florence-2 gehört zum VLM-as-Detector-Tier von LibreYOLO. Dieses besitzt eine
eigene Factory und eine getrennte Produktoberfläche neben den
Checkpoint-basierten Familien. Es benötigt das Zusatzpaket `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen
und lokal zwischengespeichert. LibreYOLO lädt die erneute Veröffentlichung des
Checkpoints durch florence-community statt des ursprünglichen Repositorys
`microsoft/Florence-2-*` herunter. Den Grund dafür findest du unter
Lizenzierung.

<code-tabs name="predict" />

Diese Familie wird über die Factory `LibreVLM()` und nicht über `LibreYOLO()`
geladen. VLM-Familien deklarieren keinen Checkpoint-Loader. Das auf anderen
Modellseiten beschriebene Routing über die Dateiendung gilt daher nicht.
`set_classes()` legt das Vokabular fest, nach dem Florence-2 im Bild suchen
soll. Es bleibt wirksam, bis du es erneut festlegst, und gilt somit für jeden
späteren Aufruf von `predict()` oder `track()`. Das zurückgegebene `Results`
enthält `boxes` in derselben Form wie jede andere Familie. Jede Erkennung trägt
jedoch dieselbe Platzhalter-Confidence. Die Filterung mit `conf` ist daher eine
Alles-oder-nichts-Entscheidung statt einer Rangfolge. `iou` hat keine Wirkung:
Der Florence-2-Wrapper erstellt die Erkennungsliste direkt aus der geparsten
Ausgabe des Aufgaben-Tokens und führt keinen Deduplizierungsschritt aus.
`chat()` löst hier `NotImplementedError` aus, weil Florence-2 mit dem
Aufgaben-Token `<OPEN_VOCABULARY_DETECTION>` statt mit einem Chat-Template
gesteuert wird. Die CLI von LibreYOLO unterstützt diesen Tier nicht. Es gibt
keine Form `libreyolo predict model=...` dafür. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt zwei Größen, Florence-2-base und Florence-2-large, beide mit 768 px.
Sie werden als `LibreVLM("florence-2-base")` oder
`LibreVLM("florence-2-large")` geladen. LibreYOLO hat noch keinen Benchmark
zum Vergleich ihrer Accuracy veröffentlicht.

LibreYOLO trainiert, validiert oder exportiert Florence-2 nicht. `train()`,
`val()` und `export()` lösen bei jeder Familie dieses Tiers den Fehler
`NotImplementedError` aus (siehe Support-Tier oben). Führe das Fine-Tuning von
Florence-2 upstream durch und lade die erzeugten Gewichte, wenn du ein fest
integriertes eigenes Vokabular benötigst. Prüfe die Ausgabe von `predict()`
visuell statt mit einem COCO-artigen Validierungsdurchlauf, weil jede Erkennung
dieselbe Platzhalter-Confidence trägt.

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
