---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'OV-DEIM in LibreYOLO: Open-Vocabulary-Erkennung'
description: >-
  Nutze OV-DEIM in LibreYOLO für Open-Vocabulary-Erkennung im DETR-Stil in
  Echtzeit. Installiere das Zusatzpaket openvocab und sage mit einem
  Freitextvokabular vorher.
lead: >-
  OV-DEIM ist ein Open-Vocabulary-Objektdetektor im DETR-Stil, der
  Decoder-Queries mit Text-Embeddings eines mitgelieferten
  MobileCLIP-Text-Towers abgleicht. LibreYOLO portiert ihn nativ als reine
  Vorhersagefamilie im Tier der Open-Vocabulary-Detektoren.
keywords:
  - ov-deim
  - deimv2
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

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vokabular ersetzen
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("ov-deim-l")

        model.set_classes(["traffic light", "bicycle"])

        first = model.predict(SAMPLE_IMAGE, conf=0.3)


        # Ein zweiter set_classes()-Aufruf ersetzt das Vokabular vollständig und

        # bettet es erneut über den Text-Tower ein. Ein leeres Ergebnis ist
        gültig

        # und kein Fehler.

        model.set_classes(["giraffe"])

        second = model.predict(SAMPLE_IMAGE, conf=0.5)

        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## Installation

OV-DEIM wird über den Tier der Open-Vocabulary-Detektoren von LibreYOLO
geladen. Dafür ist das Zusatzpaket `openvocab` erforderlich:

```bash
pip install "libreyolo[openvocab]"
```

Anders als der Rest dieses Tiers ist OV-DEIM eine native LibreYOLO-Portierung
und kein Wrapper für `transformers`. Für das Modell existiert keine Klasse in
`transformers`. Dasselbe Zusatzpaket deckt jedoch die zur Vorhersage benötigten
Pakete `huggingface_hub`, `safetensors`, `regex` und `ftfy` ab.

## Vorhersage

OV-DEIM ist kein Checkpoint, den LibreYOLO über `LibreYOLO()` lädt. Stattdessen
wird er über die verwandte Factory `LibreOpenVocab` geladen. Diese lädt bei der
ersten Verwendung einen Hugging-Face-Snapshot herunter und speichert ihn unter
`weights/` zwischen.

<code-tabs name="predict" />

`set_classes()` legt ein dauerhaftes Textvokabular fest. Rufe die Methode
erneut auf, um die Liste vollständig zu ersetzen, oder überspringe sie, um die
standardmäßigen COCO-80-Labels beizubehalten. Ein leeres Ergebnis ist gültig
und kein Fehler. Jede Decoder-Query wird anhand der Kosinusähnlichkeit mit
Text-Embeddings aus einem mitgelieferten MobileCLIP-B(LT)-Text-Tower bewertet.
Diese werden online für das jeweilige Vokabular berechnet und bis zu dessen
Änderung zwischengespeichert. Beliebige Prompts funktionieren daher ohne eine
vorab berechnete Embedding-Datei.

OV-DEIM besitzt keinen Text-Token-Schwellenwert. Nur `conf` filtert Erkennungen,
und die Übergabe von `text_threshold` löst einen Fehler aus. Der Abgleich ist
eine One-to-One-Top-K-Auswahl. Es wird keine Non-Maximum Suppression ausgeführt.
`iou` wird zur API-Kompatibilität akzeptiert, gibt aber eine Warnung aus und
hat keine Wirkung. `imgsz` und `augment=True` werden direkt abgelehnt. Das
Modell besitzt eine feste Letterbox-Eingabe, und Test-Time Augmentation liegt
außerhalb des Umfangs dieses Tiers. `predict()` gibt für ein einzelnes Bild ein
`Results`-Objekt und keine Liste zurück. Übergib ein Verzeichnis, eine Bildliste
oder `stream=True` für eine Videoquelle, um mehrere zu erhalten. Für diese
Familie gibt es keinen CLI-Pfad. `libreyolo predict` lädt über `LibreYOLO()`
nur `.pt`-Checkpoints. `LibreOpenVocab`-Familien werden daher in Python
ausgeführt. Unter [Vorhersage](/docs/predict) findest du Quelltypen und
Streaming.

Jeder Aufruf von `predict()` führt außerdem den mitgelieferten
MobileCLIP-B(LT)-Text-Tower aus, um das aktuelle Vokabular einzubetten. Unter
Lizenzierung findest du die daraus entstehenden zusätzlichen Bedingungen.

## Varianten

Es gibt drei Checkpoints, `s`, `m` und `l`. `s` ist die Standardgröße dieses
Tiers, wenn keine angegeben wird. Anders als der Rest dieses Tiers ist OV-DEIM
eine native Portierung und kein Wrapper für `transformers`. LibreYOLO übernimmt
die Detektormodule unter derselben Apache-2.0-Lizenz wie der Upstream-Code und
verwendet den bereits für die DEIMv2-Familie erstellten DINOv3-Backbone-Adapter.
Das Backbone des Checkpoints `l` ist ein DINOv3-S-Fine-Tuning und steht separat
unter der DINOv3 License von Meta. Für diese Familie sind noch keine Accuracy-
oder Latenzwerte veröffentlicht.

Training, Datensatzvalidierung und Export liegen außerhalb des Umfangs dieses
Tiers: `train()`, `val()` und `export()` lösen immer `NotImplementedError` aus.
Dies ist ein reiner Vorhersage-Wrapper um einen veröffentlichten Checkpoint.

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box>

OV-DEIM ergänzt jeden Vorhersageaufruf um drei Upstream-Lizenzen: die
Detektorgewichte unter OV-DEIMs eigener CC BY-NC 4.0, den Online-Text-Tower
unter Apples Machine Learning Research Model License (nur für Forschung) und
beim Checkpoint `l` ein nachtrainiertes DINOv3-S-Backbone unter Metas DINOv3
License. Alle drei Lizenztexte werden im LibreYOLO-Gewichtsrepository
mitgeliefert.

</provenance-box>

## Zitieren

<citation-block />
