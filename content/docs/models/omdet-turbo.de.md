---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo in LibreYOLO: Zero-Shot-Erkennung in Echtzeit'
description: >-
  Nutze OMDet-Turbo in LibreYOLO für Open-Vocabulary-Erkennung in Echtzeit.
  Installiere das Zusatzpaket openvocab und sage mit einem Freitextvokabular
  vorher.
lead: >-
  OMDet-Turbo ist ein Open-Vocabulary-Objektdetektor für Echtzeit von Om AI Lab,
  der Klassen-Embeddings von einem sprachlichen Aufgaben-Prompt entkoppelt.
  LibreYOLO bindet ihn als reine Vorhersagefamilie im Tier der
  Open-Vocabulary-Detektoren ein.
keywords:
  - omdet-turbo
  - omdet
  - open-vocabulary objekterkennung
  - echtzeit erkennung
  - zero-shot objekterkennung
  - libreopenvocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Eigener NMS-Schwellenwert
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("omdet-turbo")

        model.set_classes(["traffic light", "bicycle"])


        # OMDet-Turbo ist die einzige Familie dieses Tiers, die iou=
        berücksichtigt:

        # Die eigene Nachverarbeitung erhält den Unterdrückungsschwellenwert als
        Argument,

        # standardmäßig 0.5, wenn iou= nicht gesetzt ist.

        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)

        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Installation

OMDet-Turbo wird über den Tier der Open-Vocabulary-Detektoren von LibreYOLO
geladen. Dafür ist das Zusatzpaket `openvocab` erforderlich:

```bash
pip install "libreyolo[openvocab]"
```

Dieses Zusatzpaket installiert `transformers` und `timm`, die von diesem Tier
aufgerufenen Hugging-Face-Bibliotheken. Das Swin-Backbone von OMDet-Turbo wird
über den Wrapper `TimmBackbone` von `transformers` geladen.

## Vorhersage

OMDet-Turbo ist kein Checkpoint, den LibreYOLO über `LibreYOLO()` lädt.
Stattdessen wird er über die verwandte Factory `LibreOpenVocab` geladen. Diese
lädt bei der ersten Verwendung einen Hugging-Face-Snapshot herunter und
speichert ihn unter `weights/` zwischen.

<code-tabs name="predict" />

`set_classes()` legt ein dauerhaftes Textvokabular fest. Rufe die Methode
erneut auf, um die Liste vollständig zu ersetzen, oder überspringe sie, um die
standardmäßigen COCO-80-Labels beizubehalten. Ein leeres Ergebnis ist gültig
und kein Fehler. Anders als Grounding DINO entkoppelt OMDet-Turbo seine
Klassen-Embeddings vom sprachlichen Aufgaben-Prompt. Die Nachverarbeitung von
`transformers` gibt daher Labels zurück, die ohne Phrasen-Disambiguierung
direkt auf die angefragte Klassenliste abgebildet werden.

OMDet-Turbo besitzt keinen Text-Token-Schwellenwert. Nur `conf` filtert
Erkennungen, und die Übergabe von `text_threshold` löst einen Fehler aus. Es
ist die einzige Familie dieses Tiers, die in
`post_process_grounded_object_detection` eine eigene Non-Maximum Suppression
ausführt. `iou` wird daher berücksichtigt und erzeugt nicht nur eine Warnung.
`imgsz` und `augment=True` werden direkt abgelehnt. Der Prozessor von
`transformers` übernimmt die Größenänderung, und Test-Time Augmentation liegt
außerhalb des Umfangs dieses Tiers. `predict()` gibt für ein einzelnes Bild ein
`Results`-Objekt und keine Liste zurück. Übergib ein Verzeichnis, eine Bildliste
oder `stream=True` für eine Videoquelle, um mehrere zu erhalten. Für diese
Familie gibt es keinen CLI-Pfad. `libreyolo predict` lädt über `LibreYOLO()`
nur `.pt`-Checkpoints. `LibreOpenVocab`-Familien werden daher in Python
ausgeführt. Unter [Vorhersage](/docs/predict) findest du Quelltypen und
Streaming.

## Varianten

Es gibt einen Checkpoint, `t`, die einzige Größe des Tiers. Er spiegelt
`omlab/omdet-turbo-swin-tiny-hf` bei einer festgeschriebenen Upstream-Revision
über `OmDetTurboForObjectDetection` aus `transformers`. Die gespiegelte
Gewichtsdatei ist bytegleich mit diesem Upstream-Snapshot. Für diese Familie
sind noch keine Accuracy- oder Latenzwerte veröffentlicht.

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
