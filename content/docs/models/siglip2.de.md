---
title: SigLIP2
families:
  - siglip2
seo_title: 'SigLIP2 in LibreYOLO: Zero-Shot-Klassifizierung und Embeddings'
description: >-
  Verwende SigLIP2 in LibreYOLO für die Zero-Shot-Bildklassifizierung sowie
  Bild- und Text-Embeddings mit Sigmoid-Multi-Label-Bewertung. Kein Training
  erforderlich.
lead: >-
  SigLIP2 ist ein Dual-Tower-Modell, das ein Bild anhand von Text-Prompts
  bewertet. Dabei verwendet es für jede Klasse ein unabhängiges Sigmoid statt
  eines gemeinsamen Softmax über einen festen Label-Satz. LibreYOLO unterstützt
  es für die Zero-Shot-Klassifizierung sowie Bild- und Text-Embeddings, ganz
  ohne Trainingsschritt.
keywords:
  - SigLIP2
  - SigLIP 2
  - Zero-Shot-Klassifizierung
  - Bild-Embedding
  - Text-Embedding
  - Open Vocabulary
  - mehrsprachige Bildklassifizierung
  - Sigmoid Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Ohne Aufruf von set_classes() verwendet die CLI-Vorhersage die 1.000

        # ImageNet-Klassennamen, die das Modell standardmäßig lädt.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Sigmoid-Bewertung für mehrere Labels
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)

        r = model(SAMPLE_IMAGE)


        # Unabhängige Wahrscheinlichkeiten je Klasse: Mehrere oder auch keine

        # können gleichzeitig hoch ausfallen. Softmax (der Standard) normiert
        sie

        # stattdessen wie bei LibreCLIP zu einer Single-Label-Verteilung.

        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Bild- und Text-Embeddings
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Beide sind L2-normalisiert, daher entspricht ein einfaches
        Skalarprodukt der Kosinusähnlichkeit.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")


        # data ist ein ImageFolder-Stammverzeichnis mit einem train/-Split;

        # dessen Ordnernamen werden für diesen Lauf zu
        Zero-Shot-Klassen-Prompts.

        metrics = model.val(data="imagenette160")


        print(metrics["metrics/accuracy_top1"])

        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Die aktuellen Labels aus set_classes() und die Eingabeauflösung werden
        # in den Graphen eingebettet. Exportiere nach jeder Änderung erneut.
        # multi_label muss beim Export False (der Standard) sein.
    - label: CLI
      language: bash
      code: |
        # Ohne Aufruf von set_classes() werden hier die standardmäßigen 1.000
        # ImageNet-Klassen eingebettet, die das Modell lädt.
        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Embedding-Export
      language: python
      code: >
        from libreyolo import LibreYOLO


        # task="embed" zeichnet nur den Bild-Tower auf; Klassen sind nicht
        erforderlich.

        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Installation

SigLIP2 benötigt ein eigenes Extra, das das SentencePiece-Paket für seinen mehrsprachigen Tokenizer installiert.

```bash
pip install "libreyolo[siglip2]"
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

`set_classes()` ist die zentrale Funktion, die daraus einen Open-Vocabulary-Klassifikator macht. Sie setzt jedes Label in alle Prompt-Vorlagen ein, codiert und mittelt die Ergebnisse und speichert die resultierende `[K, D]`-Matrix als Klassifikationskopf zwischen. Dadurch wird sie nicht für jedes Bild neu berechnet. Rufe die Funktion jederzeit erneut auf, um die Klassen zu ändern. Ohne Aufruf lädt LibreSigLIP2 die bereits festgelegten 1.000 Klassennamen von ImageNet-1k.

SigLIP bewertet jede Klasse unabhängig: `logit = scale * (image . text) + bias`. Standardmäßig wird diese Menge von Logits dennoch durch ein Softmax geleitet. So entsteht eine Single-Label-Verteilung, deren Verhalten für `top1` und `top5` dem von LibreCLIP entspricht. Wenn du `multi_label=True` an `set_classes()` oder beim Erstellen des Modells übergibst, verwendet das Modell stattdessen unabhängige Sigmoid-Wahrscheinlichkeiten. Dadurch können auf demselben Bild mehrere Klassen oder gar keine Klasse eine hohe Bewertung erzielen. Der Tokenizer ist ein mehrsprachiges SentencePiece-Modell mit Gemma-Vokabular, sodass Klassennamen in anderen Sprachen genauso funktionieren wie englische.

Mit `task="embed"` gibt die Vorhersage statt Klassenwahrscheinlichkeiten einen L2-normalisierten Bildvektor je Eingabe zurück. `embed_text()` liefert normalisierte Textzeilen im selben Vektorraum, sodass ihr einfaches Skalarprodukt der Kosinusähnlichkeit entspricht. `iou` hat bei keiner der beiden Aufgaben eine Wirkung, da es keinen NMS-Schritt gibt. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Validierung

`val()` liest die Namen der Klassenordner unter dem `train/`-Split eines ImageFolder-Datensatzes, ruft damit `set_classes()` auf und misst anschließend die Zero-Shot-Genauigkeit für Top-1 und Top-5 unter Softmax-Bewertung. Die Genauigkeit hängt davon ab, wie gut sich die Klassennamen als Prompts eignen, nicht von einer Aktualisierung der Gewichte, da kein Training stattfindet. Die Validierung gilt nur für `task="classify"`; für `task="embed"` gibt es keinen Datensatz-Validator.

<code-tabs name="val" />

## Export

<export-matrix />

Beim Export wird der aktuelle Zustand des Modells in einen festen Graphen eingebettet. Für `task="classify"` werden die zuletzt mit `set_classes()` festgelegten Labels und die Auflösung zum Exportzeitpunkt in eine abschließende lineare Schicht mit dem gelernten Skalierungsfaktor und Bias übernommen. Der exportierte Graph ist damit ein gewöhnlicher `[B, K]`-Bildklassifikator ohne Text-Tower und Tokenizer. Exportiere erneut, nachdem du die Klassen oder die Größe geändert hast. Der Export im Modus `multi_label=True` ist nicht implementiert. Setze den Wert zuerst wieder auf `False`. Beim Export mit `task="embed"` wird nur der Bild-Tower aufgezeichnet. Beide Varianten benötigen ONNX-Opset 14 oder höher, den der Exporter standardmäßig festlegt.

<code-tabs name="export" />

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien. Beide wurden aus Googles Apache-2.0-Checkpoints `siglip2-base-patch16-256` und `siglip2-so400m-patch14-384` konvertiert, nicht aus einem COCO-Trainingslauf.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
