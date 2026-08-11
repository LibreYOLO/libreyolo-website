---
title: Bildrestaurierung
seo_title: Bildrestaurierung und Hochskalierung in LibreYOLO
description: >-
  Entrausche, entschärfe und skaliere Bilder in LibreYOLO hoch. Sage ein
  restauriertes RGB-Bild vorher, trainiere NAFNet auf gepaarten Daten und lies
  die PSNR- und SSIM-Keys.
lead: >-
  Die Bildrestaurierung nimmt ein degradiertes Bild und gibt ein sauberes
  zurück. LibreYOLO stellt sie als restore-Task bereit, der Entrauschen,
  Entschärfen und Super-Resolution hinter einem einzigen Ausgabevertrag abdeckt:
  ein RGB-Bild hinein, ein RGB-Bild heraus.
keywords:
  - bildrestaurierung python
  - bild entrauschen modell
  - bild hochskalieren python
  - deblurring modell
  - PSNR SSIM validierung
last_verified: 1.5.0
snippets:
  predict:
    - label: Ein Bild hochskalieren
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Der kompakte 4x-Generator; tile begrenzt den Speicherbedarf.
        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")
        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)

        result.restored.save("upscaled.png")
        print(result.restored.array.shape)   # 4x der Eingabe je Achse
    - label: Ein Bild entrauschen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Auf SIDD-Realbildrauschen trainiert; Ausgabe in Eingabegröße.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
        print(result.restore_scale)   # 1: skaliert nicht hoch
  train:
    - label: NAFNet auf gepaarten Bildern nachtrainieren
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: Die Herkunft im Checkpoint festhalten
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradation und dataset werden zur Herkunftsdokumentation in den
        # Checkpoint geschrieben; am Training nehmen sie nicht teil.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Validieren und die Metrik-Keys lesen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() liefert ein schlichtes dict, kein Objekt.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # fitness
        print(metrics["metrics/SSIM"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz wird fest in den Graph geschrieben, gib also die Größe an,
        # die dein Deployment dem Modell wirklich zuführt.
        model.export(format="onnx", imgsz=256)
    - label: Die exportierte Datei ausführen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Die Factory routet über die Dateiendung: Ein exportiertes Artefakt
        # lädt wie jeder Checkpoint und liefert dasselbe Results-Objekt.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## Definition

Der `restore`-Task bildet ein Bild auf ein anderes Bild ab. Entrauschen,
Entschärfen und Super-Resolution sind hier alle derselbe Task, weil sie einen
Vertrag teilen: Das Modell nimmt ein RGB-Bild und liefert ein RGB-Bild, und die
Degradation, die es rückgängig zu machen gelernt hat, ist eine Eigenschaft des
Checkpoints und nicht der API.

Eine Vorhersage füllt `result.restored`, ein `RestoredImage`-Payload mit einem
`(H, W, 3)` uint8-RGB-Array. `.array` liefert es als NumPy und `.save(path)`
schreibt es auf die Festplatte. `result.restore_scale` hält den Skalierungsfaktor
fest, den die Ausgabefläche trägt, und der ist `1` bei einem Checkpoint, der die
Auflösung erhält. `result.boxes` bleibt leer, deshalb werden `conf`, `iou` und
`max_det` zwar zur Signaturparität akzeptiert, haben aber keine Wirkung, und
`save=True` schreibt das restaurierte Bild direkt statt eines annotierten Fotos.

## Modelle

Drei Familien bedienen `restore`, aufgeteilt nach der Degradation, die sie
rückgängig machen.

[NAFNet](/docs/models/nafnet) ist der Entrauscher und die einzige
Restore-Familie, die LibreYOLO trainieren kann. Ihre Architektur ersetzt die
nichtlinearen Aktivierungen eines UNet-Blocks durch elementweise Multiplikation,
und der veröffentlichte Checkpoint ist auf SIDD-Realbildrauschen trainiert. Die
Ausgabe bleibt bei der Eingabeauflösung.

[Real-ESRGAN](/docs/models/real-esrgan) ist der praxistaugliche Upscaler: drei
Checkpoints, trainiert gegen synthetische Degradationen statt nur gegen
bikubisches Herunterskalieren, mit 4x, 2x und einem kleineren, schnelleren
4x-Generator für niedrigere Latenz.

[SwinIR](/docs/models/swinir) skaliert 4x hoch, mit einem
Swin-Transformer-Backbone, in drei Größen, die den offiziellen
Lightweight-Generator und zwei Real-World-Generatoren abdecken.

## Vorhersage

Die Gewichte werden beim ersten Aufruf von Hugging Face geladen und lokal
zwischengespeichert.

<code-tabs name="predict" />

Die Restaurierung läuft auf der eigenen Auflösung des Quellbildes statt auf
einer festen Netzfläche und paddet nur auf den Downsample-Faktor des Netzes,
deshalb skalieren sowohl Zeit als auch Speicher mit der Pixelzahl deiner
Eingabe. `tile` teilt den Forward-Pass in überlappende Kacheln und blendet die
Nähte wieder zusammen, und `tile_pad` ist der Rand, der um jede Kachel gelegt
wird, bevor sie wieder herausgeschnitten wird; beides sind
Python-Schlüsselwortargumente. Siehe [Vorhersage](/docs/predict) für Quellen,
Streaming und den Umgang mit Ergebnissen.

## Datensatzformat

Die Restaurierung paart jedes degradierte Eingabebild mit einem sauberen
Zielbild in exakt derselben Auflösung, zugeordnet über den Dateistamm.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` und `names` sind Schema-Platzhalter; ein Restore-Modell liefert
`Results.restored`, keine Detektionen. `degradation` und `dataset` sind
optionale Herkunftsangaben. `target_stem_suffix` deckt Datensätze ab, die das
saubere Bild anders benennen als sein degradiertes Gegenstück. Die Validierung
behält die native Auflösung und paddet nur so weit, dass sich ein Batch stapeln
lässt, die Metriken werden also auf der ursprünglichen Fläche berechnet. Siehe
[Datensatzformate](/docs/reference/dataset-formats) für den vollständigen
Vertrag.

## Training

NAFNet ist die einzige Restore-Familie mit einer Trainingsimplementierung.
`Real-ESRGAN.train()` und `SwinIR.train()` lösen beide `NotImplementedError`
aus: Diese Checkpoints stammen aus GAN-Training über synthetische
Degradations-Pipelines, und der Trainer für gepaarte Restaurierung liefe, ohne
dieses Rezept zu reproduzieren.

<code-tabs name="train" />

Der Trainer nimmt gekoppelte Ausschnitte des Eingabe-Ziel-Paares, damit beide
Seiten ausgerichtet bleiben. Siehe [Training](/docs/train) für Datensätze,
Multi-GPU und Logger, und die [NAFNet-Seite](/docs/models/nafnet) für die
Standardwerte dieser Familie und das Pooling zur Inferenzzeit, das sie während
des Trainings abhängt.

## Validierung

`val()` vergleicht die restaurierte Ausgabe mit dem sauberen Ziel, in RGB, auf
der ursprünglichen Fläche, ohne Randbeschnitt und ohne Skalierung.

<code-tabs name="val" />

`metrics/PSNR` ist der Spitzen-Signal-Rausch-Abstand in Dezibel und zugleich
`fitness`, die Zahl, die die Auswahl des besten Checkpoints liest.
`metrics/SSIM` ist die strukturelle Ähnlichkeit in `[0, 1]`, berechnet mit einem
11x11-Gauß-Fenster bei Sigma 1.5 und über die drei Farbkanäle gemittelt. Bei
beiden ist größer besser.

## Export

Ein exportiertes Restore-Modell lädt über `LibreYOLO()` anhand seiner
Dateiendung zurück, deshalb verhält sich eine `.onnx`- oder `.engine`-Datei wie
ein Checkpoint und liefert dieselben `Results`, wobei `restored` das
Ausgabebild trägt.

<code-tabs name="export" />

Der Restore-Export schreibt die räumliche Auflösung fest in den Graph, gib also
das `imgsz` an, das dein Deployment dem Modell wirklich zuführt. Bei NAFNet muss
diese Größe durch den Downsample-Faktor des Netzes teilbar sein, und unter
`dynamic=True` bleibt allein die Batch-Dimension dynamisch. Bei Real-ESRGAN und
SwinIR fällt ein weggelassenes `imgsz` auf eine kleine interne Patchgröße zurück
statt auf deine Arbeitsauflösung. Die Abdeckung pro Format steht auf jeder
Modellseite und in der
[vollständigen Export-Matrix](/docs/reference/export-matrix).
[Export](/docs/export) listet die Argumente auf, die jedes Format akzeptiert.
