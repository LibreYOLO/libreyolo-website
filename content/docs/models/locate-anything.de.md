---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: Open-Vocabulary-Erkennung und Punktlokalisierung'
description: >-
  Nutze LocateAnything in LibreYOLO für Open-Vocabulary-Erkennung und
  Punktlokalisierung. Sage mit beliebigen Textlabels vorher. Training,
  Validierung und Export werden nicht unterstützt.
lead: >-
  LocateAnything ist ein Vision-Language-Grounding-Modell von NVIDIA, das
  Bounding Boxes und Punkte parallel dekodiert statt ein Koordinaten-Token nach
  dem anderen. LibreYOLO bindet es als Open-Vocabulary-Detektor und
  Punktlokalisierer ein: Jede Liste von Textlabels wird ohne festen Head oder
  Fine-Tuning zur Klassenmenge.
keywords:
  - locateanything
  - nvidia vision-language model
  - open-vocabulary-erkennung
  - punkterkennung
  - vlm grounding
  - librevlm
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Offenes Vokabular: Beliebige Wörter statt eines festen Klassen-Heads.
        Bleibt

        # über alle späteren predict()/track()-Aufrufe wirksam, bis es neu
        gesetzt wird.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Punkt-Prompting
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        # task="point" gibt pro passendem Objekt einen Punkt statt einer Box
        zurück.

        # Aufgaben auf einem geladenen Modell mit model.set_task("point")
        wechseln.

        model = LibreLocateAnything(size="3b", task="point")

        model.set_classes(["the person closest to the camera"])

        result = model(SAMPLE_IMAGE, save=True)


        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Direkter Chat
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # Der direkte Weg unterhalb der Erkennungsfunktion: freie Fragen,
        # Zählaufgaben oder beliebige Prompts außerhalb des Boxen-Wrappers.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## Installation

LocateAnything benötigt das Zusatzpaket `vlm`. Dieses installiert
`transformers` sowie die Pakete `decord`, `lmdb` und `peft`, die der entfernte
Hugging-Face-Code beim Laden importiert.

```bash
pip install "libreyolo[vlm]"
```

## Vorhersage

`LibreLocateAnything` ist eine Python-Klasse und kein `.pt`-Checkpoint. Sie
wird weder über die Factory `LibreYOLO()` geladen noch von der CLI `libreyolo`
aufgelöst. Die Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`)
erreicht diese Familie ebenfalls über einen Alias, zum Beispiel
`LibreVLM("locate-anything")`. Sie erzeugt die unten verwendete Klasse. Beim
Laden wird der eigene entfernte Modellcode von NVIDIA auf Hugging Face
heruntergeladen und ausgeführt. LibreYOLO schreibt den Download daher auf eine
feste Commit-Revision statt auf den veränderlichen Branch `main` fest und
protokolliert vor dem ersten Download einmalig einen Lizenzhinweis.

<code-tabs name="predict" />

`result.boxes` (Aufgabe `detect`) und `result.points` (Aufgabe `point`)
enthalten die geparste Ausgabe wie bei anderen Familien. Die Confidence ist
ein Platzhalter: LocateAnything gibt keinen Score pro Box aus. Jede Erkennung
erhält deshalb dieselbe konstante Confidence. `conf=` entfernt nur Zeilen
unterhalb dieser Konstante und sortiert sie nicht. Wenn du `set_classes()`
überspringst, verwendet das Vokabular standardmäßig die COCO-80-Namen. Unter
[Vorhersage](/docs/predict) findest du Quellen, Streaming und die Verarbeitung
von Ergebnissen.

## Varianten

Es gibt eine veröffentlichte Größe, 3b. Zwei Aufgaben verwenden dieselben
Gewichte: `detect` (der Standardwert) gibt Boxen zurück, `task="point"` dagegen
einen einzelnen Punkt pro passendem Objekt in `result.points`. Auf einem
bereits geladenen Modell wechselst du mit `model.set_task("point")` zwischen
ihnen. Der Benchmark-Testaufbau von LibreYOLO hat diese Familie nicht gemessen.
Es gibt daher keine veröffentlichten Accuracy-Werte für einen Vergleich.

LibreYOLO bietet diese Familie nur für Vorhersagen an. `train()`, `val()` und
`export()` lösen alle `NotImplementedError` aus. Führe ein Fine-Tuning upstream
durch und lade stattdessen das Ergebnis. Die Datensatzvalidierung wird
übersprungen, weil eine Platzhalter-Confidence den COCO-mAP irreführend machen
würde. Der Export liegt außerhalb des Umfangs eines generativen Modells ohne
tracebares State-Dictionary.

## Lizenzierung

<provenance-box>

Die NVIDIA License erlaubt Nutzung, Vervielfältigung und Änderung, beschränkt
das Modell und alle abgeleiteten Werke für alle außer NVIDIA und verbundene
Unternehmen jedoch auf nicht kommerzielle Nutzung, Forschung oder Auswertung.
Es gibt weder einen Umsatzschwellenwert noch eine bezahlte Ausnahme.
LocateAnything-3B kombiniert außerdem zwei weitere lizenzierte Komponenten:
ein Sprach-Backbone Qwen2.5-3B-Instruct unter der Qwen Research License und
einen Vision-Encoder MoonViT-SO-400M unter MIT. LibreYOLO hostet, spiegelt oder
verteilt nichts davon. `LibreLocateAnything` lädt bei der ersten Ausführung die
Gewichte und den erforderlichen entfernten Code direkt aus
`nvidia/LocateAnything-3B` auf Hugging Face, festgeschrieben auf einen Commit.

</provenance-box>

## Zitieren

<citation-block />
