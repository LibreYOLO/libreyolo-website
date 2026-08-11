---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision in LibreYOLO: 7 Aufgaben, ein Checkpoint'
description: >-
  Nutze SenseNova-Vision in LibreYOLO für Erkennung, Segmentierung, Panoptik,
  Pose, Punkte, Tiefe und OCR aus einem Prompt-basierten generativen Checkpoint.
lead: >-
  SenseNova-Vision ist ein einheitliches multimodales Modell, das
  Bildverarbeitungsaufgaben als Prompt-basierte Generierung auf einem
  gemeinsamen Decoder formuliert: Boxen, Punkte, Keypoints und OCR-Wörter werden
  als markierter Text ausgegeben, Tiefe, Masken und panoptische Karten als vom
  Decoder gerenderte Bilder. LibreYOLO lädt es über LibreVLM und unterstützt
  sieben Aufgaben mit einem 7B-Checkpoint.
keywords:
  - sensenova-vision
  - sensetime
  - einheitliches multimodales modell
  - bagel
  - prompted detection
  - dichte wahrnehmung
  - referring segmentation
  - panoptische segmentierung
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() wechselt Aufgaben auf demselben geladenen Modell.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Referring-Segmentierung und Panoptik
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # Die Segmentierung ist referring-basiert: Sie benötigt eine Zielphrase,
        keine Klassenliste.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Ohne eigenes Vokabular greift die Panoptik auf die
        COCO-Panoptik-Kategorien

        # zurück, mit denen der Checkpoint nachtrainiert wurde.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Punkte, Pose und OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Ohne gesetztes Vokabular verwendet Pose standardmäßig "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Installation

SenseNova-Vision benötigt ein eigenes Zusatzpaket. Dieses installiert
`accelerate` für die Verteilung großer Modelle, die der Checkpoint benötigt,
und auf Nicht-macOS-Plattformen `bitsandbytes` für das 4-Bit-Laden.

```bash
pip install "libreyolo[sensenova]"
```

Der Checkpoint wird unter der eigenen LibreYOLO-Organisation auf Hugging Face
gespiegelt und bei der ersten Verwendung automatisch heruntergeladen. Er steht
unter CC BY-NC 4.0 und ist nur nicht kommerziell nutzbar. Der Loader gibt diesen
Hinweis vor jedem automatischen Download aus. Weitere Informationen findest
du unter Lizenzierung.

## Vorhersage

<code-tabs name="predict" />

Jede Vorhersage ist eine Diffusionsdekodierung auf dem gemeinsamen
Bagel-MoT-Backbone. Es handelt sich daher um ein Fähigkeitsmodell und nicht um
ein Echtzeitmodell. Die Latenz pro Bild ist deutlich höher als bei einem
speziellen Detektor oder Segmentierer. `dtype="auto"` (der Standardwert) lädt
bf16 auf einer GPU mit ausreichend Speicher und greift andernfalls auf
4-Bit-NF4-Quantisierung zurück, die `bitsandbytes` benötigt. Übergib
`dtype="bf16"`, um volle Precision auf einer ausreichend großen GPU zu
erzwingen. `noise_seed=42` beim Erstellen initialisiert den Diffusions-Sampler
für reproduzierbare dichte Ausgaben. Mit `noise_seed=None` deaktivierst du das
Seeding.

Die sieben Aufgaben verwenden einen geladenen Checkpoint. Mit `set_task()`
wechselst du ohne erneutes Laden zwischen ihnen. `set_classes()` legt das
aktive Vokabular fest. Erkennung, Punkte, Pose und Panoptik akzeptieren eine
Klassenliste. Die Segmentierung ist referring-basiert und benötigt exakt die
zu isolierende Phrase. Jede Aufgabe gibt das standardmäßige `Results`-Objekt
mit einer anderen Nutzlast zurück: `boxes` für detect, `points` für point,
`boxes` und `keypoints` für pose, `ocr` für OCR, `depth_map` für depth,
`masks` für segment und `panoptic` einschließlich `segments_info` für
panoptic. Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und
die Verarbeitung von Ergebnissen.

## Checkpoints

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
