---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL: Open-Vocabulary-Erkennung in LibreYOLO'
description: >-
  Nutze LFM2-VL in LibreYOLO für die Open-Vocabulary-Objekterkennung auf dem
  Gerät. Sage mit beliebigen Textlabels voraus. Training, Validierung und Export
  werden nicht unterstützt.
lead: >-
  LFM2-VL ist ein kompaktes Vision-Language-Modell von Liquid AI für die
  Ausführung auf dem Gerät. LibreYOLO bindet es als
  Open-Vocabulary-Objektdetektor ein: Jede Liste von Textlabels wird ohne festen
  Head und ohne erforderliches Fine-Tuning zum Klassensatz.
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - vision-language-modell
  - open-vocabulary-erkennung
  - VLM
  - edge VLM
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # Offenes Vokabular: beliebige Wörter statt eines festen Klassen-Heads.
        # Gilt für alle späteren predict()/track()-Aufrufe bis zur Neubelegung.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Unverarbeiteter Chat
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # Der Ausweg unterhalb der komfortablen Erkennung: freie Fragen,
        # Zählaufgaben oder Prompts, die der Boxen-Wrapper nicht abdeckt.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## Installation

LFM2-VL benötigt das Extra `vlm`, das `transformers` für das
Chat-Template-Backbone mitinstalliert.

```bash
pip install "libreyolo[vlm]"
```

## Vorhersage

`LibreLFM2VL` ist eine Python-Klasse und kein `.pt`-Checkpoint. Sie wird nicht
über die Factory `LibreYOLO()` geladen und von der `libreyolo`-CLI nicht
aufgelöst. Die Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`)
erreicht diese Familie ebenfalls über einen Alias, zum Beispiel mit
`LibreVLM("lfm2-vl-450m")`. Sie erstellt die nachfolgend verwendete Klasse.
Die Gewichte stammen aus dem eigenen Repository von Liquid AI auf Hugging Face
und nicht aus einem LibreYOLO-Spiegel. Beim ersten Aufruf werden sie
heruntergeladen und lokal zwischengespeichert. Zuvor wird einmalig ein
Lizenzhinweis protokolliert.

<code-tabs name="predict" />

`result.boxes` enthält wie bei jeder anderen Familie die geparsten
Erkennungen. Die Confidence ist ein Platzhalter: LFM2-VL gibt keinen Score pro
Box aus. Deshalb erhält jede Erkennung denselben konstanten Confidence-Wert,
und `conf=` verwirft lediglich Zeilen unterhalb dieser Konstante, ohne sie zu
ordnen. `iou` verwirft nahezu identische Boxen derselben Klasse oberhalb der
angegebenen Überlappung. Solche Wiederholungen entstehen als Nebeneffekt des
Greedy Decoding. Dies ist kein klassenweiser NMS-Durchlauf. Ohne
`set_classes()` verwendet das Vokabular standardmäßig die COCO-80-Namen. Unter
[Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und
Ergebnisverarbeitung.

## Varianten

Es gibt zwei Größen: 450m und 1.6b. Beide stammen aus dem LFM2.5-VL-Release von
Liquid AI und wurden für das Deployment auf dem Gerät entwickelt. Der
Benchmark-Harness von LibreYOLO hat diese Familie noch nicht vermessen. Daher
gibt es keine veröffentlichten Accuracy-Werte für einen Vergleich. Wähle eine
Größe passend zu deinem eigenen Rechenbudget.

LibreYOLO stellt diese Familie ausschließlich für Vorhersagen bereit.
`train()`, `val()` und `export()` lösen jeweils `NotImplementedError` aus:
Führe das Fine-Tuning im Upstream-Projekt durch und lade stattdessen das
Ergebnis. Die Datensatzvalidierung wird übersprungen, weil eine
Platzhalter-Confidence die COCO-mAP irreführend machen würde. Der Export liegt
bei einem generativen Modell ohne nachverfolgbares State Dict außerhalb des
Funktionsumfangs.

## Lizenzierung

<provenance-box>

Die LFM Open License v1.0 erlaubt kommerzielle Nutzung, Vervielfältigung und
Änderung, jedoch nur unterhalb einer jährlichen Umsatzgrenze von 10 Millionen
US-Dollar. Eine juristische Person, die diese Grenze erreicht oder
überschreitet, erhält durch diese Vereinbarung überhaupt keine Lizenz für die
kommerzielle Nutzung und muss sich direkt an Liquid AI wenden. Qualifizierte
gemeinnützige Organisationen sind bei nichtkommerzieller Nutzung oder
Forschungsnutzung von der Grenze ausgenommen. LibreYOLO liefert keinen
LiquidAI-Quellcode aus, da das Modell über die Apache-2.0-lizenzierte Bibliothek
`transformers` geladen wird. Es hostet oder verteilt auch die Gewichte nicht:
`LibreLFM2VL` lädt beim ersten Aufruf die passende Größe direkt aus dem eigenen
Repository von Liquid AI auf Hugging Face herunter und protokolliert zuvor
einmalig einen Hinweis.

</provenance-box>

## Zitieren

<citation-block />

