---
title: Bildklassifikation
seo_title: Bildklassifikation in LibreYOLO
description: >-
  Labele in LibreYOLO ein ganzes Bild: die Familien, die diese Aufgabe
  bedienen, das ImageFolder-Datensatzlayout und die Aufrufe für Vorhersage,
  Training, Validierung und Export.
lead: >-
  Die Bildklassifikation weist einem ganzen Bild eine Labelverteilung zu und
  lokalisiert nichts darin. Der Task-Key ist classify.
keywords:
  - bildklassifikation python
  - bildklassifikator trainieren
  - ImageFolder datensatz
  - top-1 accuracy
  - zero-shot klassifikation
  - MIT lizenz klassifikation bibliothek
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Das Suffix -cls im Dateinamen wählt den Task, ein task-Argument
        # ist also nicht nötig.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Die ganze Verteilung
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data ist der volle (C,)-Vektor; top5/top5conf sind sortiert.
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 'Zero-Shot, ohne Training'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP bewertet das Bild gegen Text-Prompts, die Labelmenge wird

        # also beim Aufruf gesetzt statt im Checkpoint festgeschrieben.

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        model.set_classes(["a person jumping", "an empty street", "a parked
        car"])

        result = model(SAMPLE_IMAGE)


        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160 ist ein bekannter Datensatzname und lädt beim ersten
        # Aufruf. Für eigene Daten ein Verzeichnis mit train/-Split angeben.
        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")

        # val() liefert ein schlichtes dict, kein Objekt.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Die exportierte Datei nutzen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung: Ein exportiertes Artefakt
        # lädt wie ein Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Definition

Die Bildklassifikation erzeugt für das ganze Bild einen Score pro Klasse und
überhaupt keine Koordinaten. Sie beantwortet, was im Bild ist, nie wo, und genau
das trennt sie von der [Objekterkennung](/docs/tasks/object-detection).

`classify` ist der kanonische Task-Key, und das Suffix `-cls` im Dateinamen
eines Checkpoints wählt ihn aus. Dieses Suffix ist bei
Klassifikations-Familien Pflicht statt optional, `LibreResNet50.pt` wird also
nicht als Klassifikator gelesen, sondern nur `LibreResNet50-cls.pt`.

`predict()` füllt `result.probs` und lässt `boxes` leer. `.data` ist der volle
Score-Vektor, `.top1` der Index des höchsten Scores und `.top1conf` sein Wert,
`.top5` die fünf höchsten Indizes in absteigender Reihenfolge und `.top5conf`
ihre Scores. Die Indizes zeigen in `result.names`. Ein `Results`-Objekt zu
slicen kürzt `probs` nie, weil der Vektor zum Bild gehört und nicht zu einer
einzelnen Zeile.

## Modelle

Fünf Familien trainieren und sagen vorher: [ResNet](/docs/models/resnet),
[ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) und
[DINOv2](/docs/models/dinov2). Die ersten vier laufen mit dem Basispaket und
bringen veröffentlichte Gewichte mit. DINOv2 braucht
`pip install "libreyolo[rfdetr]"` und hat keinen von LibreYOLO gehosteten
Checkpoint: Es lädt das Upstream-Backbone mit einem zufällig initialisierten
linearen Head, ist also ein Ausgangspunkt für Fine-Tuning und kein fertiger
Vorhersager.

Fünf weitere sagen vorher, validieren und exportieren, aber ihr `train()` löst
`NotImplementedError` aus: [ViT](/docs/models/vit), [Swin](/docs/models/swin),
[VGG](/docs/models/vgg), [AlexNet](/docs/models/alexnet) und
[DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) und [SigLIP2](/docs/models/siglip2) klassifizieren
ohne feste Labelmenge. Sie bewerten das Bild gegen Text-Prompts, deshalb
definiert `set_classes()` die Klassen zur Aufrufzeit, und für eine neue
Labelmenge gibt es überhaupt keinen Trainingsschritt. Beide bedienen auch den
`embed`-Task.

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

`conf`, `iou` und `max_det` haben hier keine Wirkung: Es gibt keine Kandidaten,
die man schwellen oder unterdrücken könnte, nur eine Verteilung. Siehe
[Vorhersage](/docs/predict) für Quellen, Streaming und den Umgang mit
Ergebnissen.

## Datensatzformat

Die Klassifikation nutzt einen Verzeichnisbaum, keine Labeldateien und keine
YAML. `data` ist das Wurzelverzeichnis des Datensatzes.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` ist für das Training erforderlich und legt die Zuordnung von Klasse zu
Index über den sortierten Ordnernamen fest, der alphabetisch erste Ordner wird
also Klasse 0. `val/` ist für die Validierung erforderlich. Ein `test/`-Split
darf vorhanden sein, und die Standardbefehle für Training und Validierung nutzen
ihn nicht. Jeder Split außer `train` muss dieselben Klassenordnernamen enthalten
wie die erwartete Klassenmenge, und genau das lässt eine Abweichung laut
scheitern, statt sie als falsche Vorhersage zu werten. Die akzeptierten
Bildendungen sind `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` und `.tiff`.

`data` nimmt drei Dinge: einen Pfad zu einem Verzeichnis mit einem
`train/`-Split, eine `.zip`-URL oder einen der bekannten Datensatznamen,
`imagenette160` und `smoke10`, die beim ersten Aufruf geladen und
zwischengespeichert werden.

Der kanonische Loader ist `libreyolo.data.classify_dataset`.

## Training

<code-tabs name="train" />

Es gibt kein `nc` zu deklarieren: Die Klassenanzahl kommt aus den Ordnernamen
unter `train/`, und die finale lineare Schicht wird passend neu aufgebaut,
während das Backbone unverändert übernommen wird. Siehe
[Training](/docs/train) für Datensätze, Datenaugmentierung, Multi-GPU und
Logger.

## Validierung

`val()` liefert ein schlichtes Dictionary von `metrics/`-Keys, berechnet über
den `val/`-Split des Datensatz-Wurzelverzeichnisses.

<code-tabs name="val" />

`metrics/accuracy_top1` ist der Anteil der Bilder, deren höchstbewertete Klasse
die richtige ist, und die Kennzahl, die das Training nutzt, um die beste Epoche
zu wählen. `metrics/accuracy_top5` ist der Anteil, dessen richtige Klasse
irgendwo unter den fünf höchstbewerteten Klassen auftaucht, was umso weniger
aussagt, je weniger Klassen der Datensatz hat. Das Dictionary trägt außerdem
`fitness`, eine Kopie des Top-1-Werts.

## Export

<code-tabs name="export" />

Ein exportiertes Artefakt lädt über `LibreYOLO()` anhand seiner Dateiendung
zurück, deshalb verhält sich eine `.onnx`- oder `.engine`-Datei wie ein
Checkpoint und liefert dieselben `Results`. Die Formatabdeckung unterscheidet
sich je Familie; die Matrix auf jeder Modellseite wird aus dem validierten Satz
erzeugt statt von Hand getippt. Siehe
[Export und Deployment](/docs/export) für die Formate, ihre Extras und ihre
Einschränkungen.
