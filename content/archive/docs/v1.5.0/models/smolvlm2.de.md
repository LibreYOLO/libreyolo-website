---
title: SmolVLM2
families:
  - smolvlm2
seo_title: 'SmolVLM2 in LibreYOLO: Open-Vocabulary-Objekterkennung'
description: >-
  SmolVLM2 in LibreYOLO: Installiere das Apache-2.0-Vision-Language-Modell von
  Hugging Face, lege ein offenes Vokabular fest und nutze Vorhersagen oder den
  Chat.
lead: >-
  SmolVLM2 ist das kleine Vision-Language-Modell von Hugging Face. LibreYOLO
  bindet es als Open-Vocabulary-Objekterkennung ein und stellt seinen freien
  Chat direkt bereit: Übergib eine Klassenliste für die Erkennung oder stelle
  dem Modell eine Frage.
keywords:
  - SmolVLM2
  - Vision-Language-Modell
  - Open-Vocabulary-Objekterkennung
  - kleines multimodales Modell
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # Der direkte Zugriff unterhalb der komfortablen Erkennung: jede Frage,
        # nicht nur eine Abfrage nach Begrenzungsrahmen.
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## Installation

SmolVLM2 gehört zur VLM-als-Detektor-Stufe von LibreYOLO. Diese ist eine separate Produktoberfläche neben den Checkpoint-basierten Familien und besitzt eine eigene Factory. Sie benötigt das Extra `vlm`, das auch `num2words` installiert, eine Abhängigkeit des SmolVLM2-Prozessors.

```bash
pip install "libreyolo[vlm]"
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Diese Familie wird über die Factory `LibreVLM()` und nicht über `LibreYOLO()` geladen. VLM-Familien definieren keinen Checkpoint-Loader, daher gilt die auf anderen Modellseiten beschriebene Weiterleitung anhand der Dateiendung hier nicht. `set_classes()` legt das Vokabular fest, nach dem SmolVLM2 suchen soll. Die Einstellung bleibt für alle späteren Aufrufe von `predict()` und `track()` bestehen, bis du sie erneut setzt. SmolVLM2 benötigt in LibreYOLO keinen eigenen Parser. Es verwendet dieselbe Kombination aus Chat-Vorlage und JSON-Ausgabe wie der gemeinsame Standard dieser Stufe. Der Erkennungs-Prompt und das Boxformat sind daher nicht familienspezifisch. Jede Erkennung erhält denselben Platzhalter-Konfidenzwert, weshalb die Filterung mit `conf` entweder alle oder keine Ergebnisse behält, statt sie zu sortieren. `iou` wirkt sich dagegen aus: Eine spätere Box derselben Klasse wird verworfen, sobald sie eine bereits beibehaltene Box über dem Schwellenwert überlappt. Andernfalls könnte ein sich wiederholender Generator nahezu identische Boxen für dasselbe Objekt ausgeben. Über `chat()` beantwortet SmolVLM2 außerdem frei formulierte Fragen. Dies ist derselbe direkte Zugriff, der für die Factory `LibreVLM` dokumentiert ist. Die LibreYOLO-CLI deckt diese Stufe nicht ab. Es gibt dafür keine Form wie `libreyolo predict model=...`. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Die Registry enthält eine Größe: SmolVLM2-500M-Video-Instruct, geladen als `LibreVLM("smolvlm2-500m")`. SmolVLM2 ist ein schwächerer Detektor als die eigens dafür entwickelten Grounding-Modelle dieser Stufe. Der LibreYOLO-Wrapper beschreibt es als Demonstration dafür, dass eine neue Familie hier ohne spezielle Parserlogik funktionieren kann, nicht als leistungsstärkste Open-Vocabulary-Option.

LibreYOLO kann SmolVLM2 weder trainieren noch validieren oder exportieren. `train()`, `val()` und `export()` lösen für jede Familie dieser Stufe `NotImplementedError` aus (siehe die oben angegebene Support-Stufe). Führe das Fine-Tuning von SmolVLM2 im Upstream-Projekt aus und lade die resultierenden Gewichte, wenn du ein fest eingebettetes eigenes Vokabular benötigst. Prüfe die Ausgabe von `predict()` visuell statt mit einem Validierungslauf im COCO-Stil, da jede Erkennung denselben Platzhalter-Konfidenzwert trägt.

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
