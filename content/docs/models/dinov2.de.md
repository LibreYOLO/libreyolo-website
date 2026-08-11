---
title: DINOv2
families:
  - dinov2
seo_title: 'DINOv2 in LibreYOLO: segmentieren, klassifizieren und einbetten'
description: >-
  Nutze DINOv2 in LibreYOLO für semantische Segmentierung, Klassifizierung und
  Ganzbild-Embeddings auf dem DINOv2-with-Registers-Backbone. Durchgehend
  Apache-2.0.
lead: >-
  DINOv2 ist ein selbstüberwachter Vision Transformer, den Meta AI ohne Labels
  für universell einsetzbare Bild-Features trainiert hat. LibreYOLO bindet sein
  DINOv2-with-Registers-Backbone für drei Aufgaben ein: semantische
  Segmentierung, Klassifizierung und Ganzbild-Embedding.
keywords:
  - dinov2 python
  - dinov2 with registers
  - selbstüberwachtes lernen bilder
  - vision transformer
  - semantische segmentierung
  - bild embeddings
  - feature extraktion
  - meta ai dinov2
last_verified: 1.5.0
snippets:
  predict:
    - label: Semantisch
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Für diese Familie gibt es keinen von LibreYOLO gehosteten Checkpoint:
        # Dies lädt das DINOv2-with-Registers-Small-Backbone unter Apache-2.0
        # von Metas Hugging-Face-Organisation. Der dichte Head ist zufällig
        # initialisiert, bis du ihn trainierst (siehe Training unten).
        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        result = model(SAMPLE_IMAGE)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: Klassifizieren
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= ist die Klassenanzahl deines Datensatzes. Der lineare Head
        # ist zufällig initialisiert, bis du ihn trainierst.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: Einbetten
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Umgeht alle Aufgaben-Heads. Das Backbone allein reicht aus,
        # daher ist kein Fine-Tuning nötig.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), L2-normalisiert
    - label: Einen Batch einbetten
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Komfortfunktion: Führt predict() aus und stapelt alle Zeilen zu einem
        # Tensor der Form (N, D).
        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(features.shape)
  train:
    - label: Semantisch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Klassifizieren
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Semantisch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Klassifizieren
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Semantisch
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Klassifizieren
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Einbetten
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.
        Export

        # benennt die Datei nach der Aufgabe, hier LibreDINOv2s-sem.onnx.

        model = LibreYOLO("LibreDINOv2s-sem.onnx")

        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## Installation

LibreDINOv2 wird nur registriert, wenn `transformers` installiert ist. Das ist
dieselbe optionale Abhängigkeit, die RF-DETR für sein DINOv2-Backbone benötigt.
Daher ist dasselbe Zusatzpaket erforderlich.

```bash
pip install "libreyolo[rfdetr]"
```

## Vorhersage

LibreYOLO veröffentlicht keinen LibreDINOv2-Checkpoint. Erzeuge den Wrapper
direkt, statt eine Datei zu laden: `model_path=None` (der Standardwert) lädt
bei der ersten Verwendung Metas Apache-2.0-Backbone
`facebook/dinov2-with-registers-small` von Hugging Face herunter. `task=` wählt
aus, was darauf ausgeführt wird.

<code-tabs name="predict" />

`task="semantic"` und `task="classify"` fügen dem Backbone einen dichten
beziehungsweise linearen Head hinzu. Dieser Head wird zufällig initialisiert
und ist erst nach dem Training nützlich (siehe [Training](#train)).
`task="embed"` überspringt alle Heads und gibt das letzte normalisierte
CLS-Token des Backbones als eine Ganzbildzeile in `result.embeddings` zurück.
Dafür ist überhaupt kein Training erforderlich. `result.boxes` ist immer
`None`, weil keine der drei Aufgaben instanzbezogene Erkennungen erzeugt.
Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und die
Verarbeitung von Ergebnissen.

## Varianten

`size` wählt die Breite des RF-DETR-artigen Projektors auf dem Backbone und
nicht das Backbone selbst. Alle Größen verwenden denselben DINOv2-S-Encoder
(Small). Die semantische Segmentierung läuft auf dem nativen quadratischen
Patch-Raster von DINOv2. Klassifizierung und Embedding nutzen die kleinere
Klassifizierungsauflösung, mit der die lineare Probe trainiert wurde.

## Training

`task="semantic"` und `task="classify"` können beide trainiert werden.
`task="embed"` besitzt keinen klassenabhängigen Head, der angepasst werden
könnte, und löst bei einem Aufruf von `train()` den Fehler `NotImplementedError`
aus.

<code-tabs name="train" />

Die wichtigsten Schlüsselwortargumente sind hier `batch_size` und `lr`, nicht
`batch` und `lr0` wie bei den meisten anderen Familien. `batch` und `lr0`
werden weiterhin akzeptiert und darauf abgebildet. Die Übergabe beider Formen
löst jedoch einen Konfliktfehler aus. `output_dir=` (standardmäßig
`"runs/train"`) ersetzt `project=`/`name=` als bevorzugte Methode zur Ablage
eines Laufs, auch wenn die direkte Übergabe von `project=`/`name=` weiterhin
funktioniert. Unter [Training](/docs/train) findest du Datensätze,
Datenaugmentierung, Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/`-Schlüsseln zurück: mIoU und
Pixel-Accuracy für `task="semantic"`, Top-1- und Top-5-Accuracy für
`task="classify"`. `task="embed"` besitzt keine Ground Truth für eine
Bewertung und löst beim Aufruf von `val()` den Fehler `NotImplementedError`
aus.

<code-tabs name="val" />

## Export

<export-matrix />

Jede Aufgabe unterstützt eine andere, oben dargestellte Teilmenge der Formate.
Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück. [Export](/docs/export) führt die
Argumente auf, die von den einzelnen Formaten akzeptiert werden.

<code-tabs name="export" />

## Lizenzierung

<provenance-box>

Die Zeile „Gewichte“ oben nennt die gültige Lizenz Apache-2.0. Für diese Familie
wird jedoch nichts unter der LibreYOLO-Organisation auf Hugging Face erneut
veröffentlicht: LibreYOLO hostet keinen eigenen LibreDINOv2-Checkpoint.
`LibreDINOv2(model_path=None)` lädt unverändert Metas eigenes Repository
`facebook/dinov2-with-registers-small` herunter.

</provenance-box>

## Zitieren

<citation-block />
