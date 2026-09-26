---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO in LibreYOLO: Open-Set-Erkennung'
description: >-
  Nutze Grounding DINO in LibreYOLO, um beliebige textbeschriebene Objekte zu
  erkennen. Installiere das Zusatzpaket openvocab und sage mit einem
  Freitextvokabular vorher.
lead: >-
  Grounding DINO ist ein Open-Set-Objektdetektor von IDEA Research. Er bewertet
  ein Bild anhand eines Freitext-Prompts statt einer festen Klassenliste.
  LibreYOLO bindet ihn als reine Vorhersagefamilie im Tier der
  Open-Vocabulary-Detektoren ein.
keywords:
  - grounding dino python
  - open-vocabulary objekterkennung
  - open-set erkennung
  - zero-shot objekterkennung
  - textkonditionierter detektor
  - libreopenvocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Textschwellenwert
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filtert nach Box-Score, text_threshold nach dem Token-Score
        # der dekodierten Phrase. Ohne Angabe sind beide standardmäßig 0.25.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Installation

Grounding DINO wird über den Tier der Open-Vocabulary-Detektoren von LibreYOLO
geladen. Dafür ist das Zusatzpaket `openvocab` erforderlich:

```bash
pip install "libreyolo[openvocab]"
```

Dieses Zusatzpaket installiert `transformers` und `timm`, die von diesem Tier
aufgerufenen Hugging-Face-Bibliotheken.

## Vorhersage

Grounding DINO ist kein Checkpoint, den LibreYOLO über `LibreYOLO()` lädt.
Stattdessen wird er über die verwandte Factory `LibreOpenVocab` geladen. Diese
lädt bei der ersten Verwendung einen Hugging-Face-Snapshot herunter und
speichert ihn unter `weights/` zwischen.

<code-tabs name="predict" />

`set_classes()` legt ein dauerhaftes Textvokabular fest. Rufe die Methode
erneut auf, um die Liste zu ersetzen, oder überspringe sie, um die
standardmäßigen COCO-80-Labels beizubehalten. Grounding DINO dekodiert freie
Phrasen aus seiner eigenen Textausgabe und ordnet sie selbst diesem Vokabular
zu. Eine exakte normalisierte Übereinstimmung gewinnt, eine Übereinstimmung
vollständiger Tokens wird akzeptiert, und eine mehrdeutige oder nicht passende
Phrase wird verworfen statt geraten. `school bus` wird daher nie nur `bus`
oder `school` zugeordnet. Wenn ein Vokabular das Token-Limit des Text-Encoders
überschreitet, wird es in mehrere Prompts aufgeteilt. Diese werden in getrennten
Vorwärtsläufen ausgeführt und wieder zu einer Menge von Erkennungen mit der
Obergrenze `max_det` zusammengeführt.

`iou` wird zur API-Kompatibilität akzeptiert, gibt aber eine Warnung aus und
hat keine Wirkung, weil hier keine Non-Maximum Suppression ausgeführt wird.
`imgsz` und `augment=True` werden direkt abgelehnt. Der Prozessor von
`transformers` übernimmt die Größenänderung, und Test-Time Augmentation liegt
außerhalb des Umfangs dieses Tiers. `predict()` gibt für ein einzelnes Bild
ein `Results`-Objekt und keine Liste zurück. Übergib ein Verzeichnis, eine
Bildliste oder `stream=True` für eine Videoquelle, um mehrere zu erhalten. Für
diese Familie gibt es keinen CLI-Pfad. `libreyolo predict` lädt über
`LibreYOLO()` nur `.pt`-Checkpoints. `LibreOpenVocab`-Familien werden daher in
Python ausgeführt. Unter [Vorhersage](/docs/predict) findest du Quelltypen und
Streaming.

## Varianten

Es gibt zwei Checkpoints, `t` und `b`. `t` ist die Standardgröße dieses Tiers,
wenn keine angegeben wird. Beide spiegeln das offizielle Release von IDEA
Research über `GroundingDinoForObjectDetection` aus `transformers`. Sie werden
einmalig in einen von LibreYOLO gehosteten Hugging-Face-Snapshot geladen, der
die Upstream-Dateien bewahrt. Für diese Familie sind noch keine Accuracy- oder
Latenzwerte veröffentlicht.

Training, Datensatzvalidierung und Export liegen außerhalb des Umfangs dieses
Tiers: `train()`, `val()` und `export()` lösen immer `NotImplementedError` aus.
Dies ist ein reiner Vorhersage-Wrapper um einen veröffentlichten Checkpoint.

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
