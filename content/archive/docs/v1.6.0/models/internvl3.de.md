---
title: InternVL3
families:
  - internvl3
seo_title: 'InternVL3: Open-Vocabulary-Erkennung in LibreYOLO'
description: >-
  Nutze InternVL3 in LibreYOLO für die Open-Vocabulary-Objekterkennung. Sage mit
  beliebigen Textlabels voraus. Training, Validierung und Export werden nicht
  unterstützt.
lead: >-
  InternVL3 ist ein natives multimodales Large Language Model von OpenGVLab, das
  Bild und Sprache gemeinsam in einer einzigen Vortrainingsphase lernt.
  LibreYOLO bindet es als Open-Vocabulary-Objektdetektor ein: Jede Liste von
  Textlabels wird ohne festen Head und ohne erforderliches Fine-Tuning zum
  Klassensatz.
keywords:
  - InternVL3
  - InternVL
  - vision-language-modell
  - open-vocabulary-erkennung
  - VLM
  - OpenGVLab
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # Offenes Vokabular: beliebige Wörter statt eines festen Klassen-Heads.
        # Gilt für alle späteren predict()/track()-Aufrufe bis zur Neubelegung.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Unverarbeiteter Chat
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # Der Ausweg unterhalb der komfortablen Erkennung: freie Fragen,
        # Zählaufgaben oder Prompts, die der Boxen-Wrapper nicht abdeckt.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## Installation

InternVL3 benötigt das Extra `vlm`, das `transformers` für das
Chat-Template-Backbone mitinstalliert.

```bash
pip install "libreyolo[vlm]"
```

## Vorhersage

`LibreInternVL3` ist eine Python-Klasse und kein `.pt`-Checkpoint. Sie wird
nicht über die Factory `LibreYOLO()` geladen und von der `libreyolo`-CLI nicht
aufgelöst. Die Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`)
erreicht diese Familie ebenfalls über einen Alias, zum Beispiel mit
`LibreVLM("internvl3-2b")`. Sie erstellt die nachfolgend verwendete Klasse.
Die Gewichte stammen aus den eigenen `-hf`-Repositorys von OpenGVLab auf
Hugging Face und nicht aus einem LibreYOLO-Spiegel. Beim ersten Aufruf werden
sie heruntergeladen und lokal zwischengespeichert. Zuvor wird einmalig ein
Lizenzhinweis zu den zugangsbeschränkten Qwen-Gewichten protokolliert.

<code-tabs name="predict" />

`result.boxes` enthält wie bei jeder anderen Familie die geparsten
Erkennungen. Die Confidence ist ein Platzhalter: InternVL3 gibt keinen Score
pro Box aus. Deshalb erhält jede Erkennung denselben konstanten
Confidence-Wert, und `conf=` verwirft lediglich Zeilen unterhalb dieser
Konstante, ohne sie zu ordnen. `iou` verwirft nahezu identische Boxen derselben
Klasse oberhalb der angegebenen Überlappung. Solche Wiederholungen entstehen
als Nebeneffekt des Greedy Decoding. Dies ist kein klassenweiser NMS-Durchlauf.
Ohne `set_classes()` verwendet das Vokabular standardmäßig die COCO-80-Namen.
Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen,
Streaming und Ergebnisverarbeitung.

## Varianten

Es gibt drei Größen: 1b, 2b und 8b. Alle sind native `-hf`-Checkpoints von
OpenGVLab mit einem Qwen-LLM-Backbone und nicht mit der Zwei-Turm-Architektur,
die im ursprünglichen InternVL-Paper beschrieben wird. Der Benchmark-Harness
von LibreYOLO hat diese Familie noch nicht vermessen. Daher gibt es keine
veröffentlichten Accuracy-Werte für einen Vergleich. Wähle eine Größe passend
zu deinem eigenen Rechenbudget.

LibreYOLO stellt diese Familie ausschließlich für Vorhersagen bereit.
`train()`, `val()` und `export()` lösen jeweils `NotImplementedError` aus:
Führe das Fine-Tuning im Upstream-Projekt durch und lade stattdessen das
Ergebnis. Die Datensatzvalidierung wird übersprungen, weil eine
Platzhalter-Confidence die COCO-mAP irreführend machen würde. Der Export liegt
bei einem generativen Modell ohne nachverfolgbares State Dict außerhalb des
Funktionsumfangs.

## Lizenzierung

<provenance-box>

Der eigene Code von InternVL3 steht unter der MIT-Lizenz. Er ist freizügig
lizenziert und kann in kommerziellen sowie Closed-Source-Produkten eingesetzt
werden. Die von dieser Familie geladenen `-hf`-Checkpoints enthalten ein
Qwen-LLM-Backbone und werden separat unter der Qwen License von Alibaba Cloud
lizenziert: Nutzung, Änderung und Weitergabe sind kostenlos, erfordern jedoch
den Hinweis „Built with Qwen“ oder „Improved using Qwen“. Außerdem gilt für
die kommerzielle Nutzung eine Obergrenze von 100 Millionen monatlich aktiven
Nutzern. Oberhalb dieser Grenze ist eine eigene Genehmigung von Alibaba
erforderlich. LibreYOLO hostet oder verteilt diese Gewichte nicht:
`LibreInternVL3` lädt beim ersten Aufruf die passende Größe direkt von
`OpenGVLab/InternVL3-<size>-hf` auf Hugging Face herunter und protokolliert
zuvor einmalig einen Hinweis zur Qwen License.

</provenance-box>

## Zitieren

<citation-block />

