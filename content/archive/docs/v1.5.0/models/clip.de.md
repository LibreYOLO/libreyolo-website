---
title: CLIP
families:
  - clip
seo_title: 'CLIP in LibreYOLO: Zero-Shot-Klassifikation und Embeddings'
description: >-
  Nutze CLIP in LibreYOLO für Zero-Shot-Bildklassifikation und
  Bild-/Text-Embeddings. Kein Training: set_classes() legt den Label-Satz zur
  Laufzeit fest.
lead: >-
  CLIP ist ein Dual-Tower-Modell, das ein Bild gegen Text-Prompts bewertet statt
  gegen einen festen Label-Satz. LibreYOLO unterstützt es für
  Zero-Shot-Klassifikation und Bild-/Text-Embeddings, ganz ohne
  Trainingsschritt.
keywords:
  - CLIP
  - OpenCLIP
  - zero-shot klassifikation
  - bild embedding python
  - bilder per text suchen
  - open vocabulary erkennung
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Ohne set_classes()-Aufruf nutzt predict im CLI die 1000 ImageNet-

        # Klassennamen, mit denen das Modell standardmäßig lädt.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Bild- und Text-Embedding
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        image_embed = model(SAMPLE_IMAGE).embeddings.data
        text_embed = model.embed_text("a photo of a forklift")

        # Beide sind L2-normalisiert, ein Skalarprodukt ist Kosinusähnlichkeit.
        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data ist ein ImageFolder-Wurzelverzeichnis mit train/-Split; seine
        # Ordnernamen werden zu den Zero-Shot-Prompts dieses Laufs.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Die aktuellen set_classes()-Labels und die Eingabeauflösung werden
        # in den Graphen eingebrannt. Nach einer Änderung neu exportieren.
    - label: CLI
      language: bash
      code: |
        # Hier kein set_classes()-Aufruf, also werden die 1000 ImageNet-
        # Standardklassen eingebrannt, mit denen das Modell lädt.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Embedding-Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" erfasst per Tracing nur den Bild-Tower, keine Klassen.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Installation

CLIP braucht ein eigenes Extra, das die Pakete mitbringt, die sein mitgelieferter BPE-Tokenizer nutzt, um exakte Token-IDs zu reproduzieren.

```bash
pip install "libreyolo[clip]"
```

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal zwischengespeichert.

<code-tabs name="predict" />

`set_classes()` ist das eine Primitiv, das daraus einen Open-Vocabulary-Klassifikator macht: Es rendert jedes Label in jede Prompt-Vorlage, kodiert die Ergebnisse, mittelt sie und legt die resultierende `[K, D]`-Matrix als Classifier-Head im Cache ab, damit sie nicht pro Bild neu berechnet wird. Rufe es erneut auf, um die Klassen jederzeit zu ändern. Ohne Aufruf lädt LibreCLIP mit den bereits gesetzten 1000 Klassennamen von ImageNet-1k.

Mit `task="embed"` liefert die Vorhersage pro Eingabe einen L2-normalisierten Bildvektor statt Klassenwahrscheinlichkeiten, und `embed_text()` liefert normalisierte Textzeilen im selben Vektorraum, sodass ein einfaches Skalarprodukt zwischen beiden die Kosinusähnlichkeit ist. `iou` hat auf keine der beiden Aufgaben eine Wirkung; es gibt keinen NMS-Schritt. Siehe [Vorhersage](/docs/predict) für Quellen, Streaming und den Umgang mit den Ergebnissen.

## Validierung

`val()` liest die Namen der Klassenordner unter dem `train/`-Split eines ImageFolder-Datensatzes, ruft damit `set_classes()` auf und misst dann die Zero-Shot-Accuracy für Top-1 und Top-5. Die Accuracy hängt davon ab, wie sich die Klassennamen als Prompts lesen, nicht von einem Update der Gewichte, denn es gibt nichts zu trainieren. Die Validierung deckt nur `task="classify"` ab; für `task="embed"` gibt es keinen Validator für Datensätze.

<code-tabs name="val" />

## Export

<export-matrix />

Der Export brennt den aktuellen Zustand des Modells in einen festen Graphen ein. Bei `task="classify"` landen die zuletzt von `set_classes()` gesetzten Labels und die Auflösung zum Zeitpunkt des Exports in einer finalen linearen Schicht, sodass der exportierte ONNX- oder TensorRT-Graph ein gewöhnlicher `[B, K]`-Bildklassifikator ohne Text-Tower und ohne Tokenizer ist; exportiere erneut, nachdem du entweder die Klassen oder die Größe geändert hast. Der Export mit `task="embed"` erfasst per Tracing nur den Bild-Tower. Beide brauchen ONNX-Opset 14 oder höher, was der Exporter standardmäßig setzt.

<code-tabs name="export" />

## Checkpoints

Jede veröffentlichte Gewichtsdatei dieser Familie. Beide sind aus den auf LAION-2B trainierten Checkpoints von OpenCLIP konvertiert (`ViT-B-32` und `ViT-B-16`), nicht aus einem COCO-Trainingslauf.

<checkpoint-table />

Die Trainingsdaten von LAION-2B haben eine dokumentierte Vorgeschichte mit CSAM-Inhalten (Stanford Internet Observatory, Dezember 2023). LAION hat inzwischen Re-LAION veröffentlicht, eine bereinigte Neuauflage; bevorzuge, wo verfügbar, von Re-LAION abgeleitete Checkpoints, wenn du diese Gewichte weiter hostest.

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
