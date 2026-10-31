---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 in LibreYOLO: Zero-Shot-Objekterkennung'
description: >-
  Nutze OWLv2 in LibreYOLO, um beliebige textbeschriebene Objekte zu erkennen.
  Installiere das Zusatzpaket openvocab und sage mit einem Freitextvokabular
  vorher.
lead: >-
  OWLv2 ist ein Open-Vocabulary-Objektdetektor von Google Research, der
  Bildregionen anhand von Text-Embeddings aus einem CLIP-artigen Encoder
  bewertet. LibreYOLO bindet ihn als reine Vorhersagefamilie im Tier der
  Open-Vocabulary-Detektoren ein.
keywords:
  - owlv2 python
  - owl-vit
  - open-vocabulary objekterkennung
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

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Standardvokabular
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        # Ohne set_classes() bleibt das standardmäßige COCO-80-Vokabular des
        Tiers erhalten.

        model = LibreOpenVocab("owlv2-l14")

        result = model.predict(SAMPLE_IMAGE, conf=0.1)

        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Installation

OWLv2 wird über den Tier der Open-Vocabulary-Detektoren von LibreYOLO geladen.
Dafür ist das Zusatzpaket `openvocab` erforderlich:

```bash
pip install "libreyolo[openvocab]"
```

Dieses Zusatzpaket installiert `transformers` und `timm`, die von diesem Tier
aufgerufenen Hugging-Face-Bibliotheken.

## Vorhersage

OWLv2 ist kein Checkpoint, den LibreYOLO über `LibreYOLO()` lädt. Stattdessen
wird er über die verwandte Factory `LibreOpenVocab` geladen. Diese lädt bei der
ersten Verwendung einen Hugging-Face-Snapshot herunter und speichert ihn unter
`weights/` zwischen.

<code-tabs name="predict" />

`set_classes()` legt ein dauerhaftes Textvokabular fest. Rufe die Methode
erneut auf, um die Liste zu ersetzen, oder überspringe sie, um die
standardmäßigen COCO-80-Labels beizubehalten. Jedes Label wird vor der Übergabe
an den Text-Tower in ein festes Prompt-Template eingebettet. Das entspricht
dem Training von `Owlv2ForObjectDetection` aus `transformers`.

OWLv2 besitzt keinen Text-Token-Schwellenwert. Nur `conf` filtert Erkennungen,
und die Übergabe von `text_threshold` löst einen Fehler aus. `iou` wird zur
API-Kompatibilität akzeptiert, gibt aber eine Warnung aus und hat keine Wirkung,
weil hier keine Non-Maximum Suppression ausgeführt wird. `imgsz` und
`augment=True` werden direkt abgelehnt. Der Prozessor von `transformers`
übernimmt die Größenänderung, und Test-Time Augmentation liegt außerhalb des
Umfangs dieses Tiers. `predict()` gibt für ein einzelnes Bild ein
`Results`-Objekt und keine Liste zurück. Übergib ein Verzeichnis, eine Bildliste
oder `stream=True` für eine Videoquelle, um mehrere zu erhalten. Für diese
Familie gibt es keinen CLI-Pfad. `libreyolo predict` lädt über `LibreYOLO()`
nur `.pt`-Checkpoints. `LibreOpenVocab`-Familien werden daher in Python
ausgeführt. Unter [Vorhersage](/docs/predict) findest du Quelltypen und
Streaming.

## Varianten

Es gibt zwei Checkpoints, `b16` (Base, Patch-Größe 16) und `l14` (Large,
Patch-Größe 14). `b16` ist die Standardgröße dieses Tiers, wenn keine angegeben
wird. Beide spiegeln das offizielle Release von Google Research über
`Owlv2ForObjectDetection` aus `transformers`. Sie werden einmalig in einen von
LibreYOLO gehosteten Hugging-Face-Snapshot geladen, der die Upstream-Dateien
bewahrt. Für diese Familie sind noch keine Accuracy- oder Latenzwerte
veröffentlicht.

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
