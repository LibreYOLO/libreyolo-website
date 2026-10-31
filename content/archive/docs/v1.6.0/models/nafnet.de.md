---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: Entrauschen, Training und Export unter MIT'
description: >-
  Nutze NAFNet in LibreYOLO für Bildentrauschen und -wiederherstellung.
  Installiere, sage vorher, trainiere, validiere und exportiere den
  MIT-lizenzierten SIDD-Checkpoint.
lead: >-
  NAFNet ist ein faltungsbasiertes Netzwerk zur Bildwiederherstellung, das die
  nichtlinearen Aktivierungsfunktionen eines typischen UNet-Blocks entfernt und
  durch elementweise Multiplikation ersetzt. LibreYOLO unterstützt eine Aufgabe,
  Wiederherstellung, mit einem veröffentlichten Checkpoint zum Entrauschen
  realer Bilder, der auf SIDD trainiert wurde.
keywords:
  - nafnet
  - bildwiederherstellung
  - bilder entrauschen
  - bildunschärfe entfernen
  - nonlinear activation free network
  - sidd
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Wiederhergestelltes Bild speichern
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Checkpoint-Herkunft
      language: python
      code: |
        from libreyolo import LibreYOLO

        # degradation und dataset werden im gespeicherten Checkpoint vermerkt;
        # sie ändern nicht, was trainiert wird.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() gibt ein einfaches dict zurück, kein Objekt
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model("noisy.jpg")


        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Installation

NAFNet benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Das zurückgegebene `Results`-Objekt enthält für diese Familie ein Feld,
`restored`, ein dichtes HWC-uint8-RGB-Bild auf der ursprünglichen Arbeitsfläche.
Es gibt keine Boxen zum Durchlaufen. `save=True` schreibt das wiederhergestellte
Bild direkt auf den Datenträger, statt eine Annotation über die Eingabe zu
zeichnen. `conf`, `iou` und `max_det` werden zur Signaturparität mit allen
anderen Familien akzeptiert, haben aber keine Wirkung. Die Wiederherstellung
erzeugt keine zu filternden Erkennungen. Unter [Vorhersage](/docs/predict)
findest du Quellen, Streaming und die Verarbeitung von Ergebnissen.

## Varianten

Diese Architektur besitzt zwei Breiten: `s` (Breite 32) und `l` (Breite 64),
beide um einen Trainings-Patch von 256 px herum aufgebaut. Vorhersage und
Validierung laufen unabhängig von der Größe mit der nativen Bildauflösung und
füllen nur bis zum Downsampling-Faktor des Netzwerks auf. Derzeit ist nur die
Breite `l` veröffentlicht, als auf SIDD trainierter Checkpoint zum Entrauschen
realer Bilder.

## Training

NAFNet wird mit deinen eigenen gepaarten beeinträchtigten und sauberen Bildern
nachtrainiert. Die Datensatz-YAML verweist auf einen Ordner
`inputs/<split>/` mit beeinträchtigten Bildern und einen Ordner
`targets/<split>/` mit sauberen Zielbildern, die anhand des Dateinamens ohne
Endung zugeordnet werden. `degradation` und `dataset` sind optionale Strings,
die zur Herkunftsangabe im gespeicherten Checkpoint vermerkt werden. Sie
beeinflussen das Training nicht.

<code-tabs name="train" />

Ohne Änderungen läuft der Trainer 100 Epochen mit AdamW bei `lr0=1e-3`, einem
Batch von 16, Zuschnitten von 256 px und Early Stopping nach 50 Epochen ohne
Verbesserung des PSNR. Für diese Familie gibt es keinen LoRA-Pfad:
`lora=True` löst einen Fehler aus, statt zu laufen, weil `NAFNetTrainer` kein
Adapter-Fine-Tuning aktiviert.

Während des Trainings verwendet das Netzwerk einfaches globales Average
Pooling. Das nur für die Inferenz verwendete fensterbasierte lokale Pooling
von NAFNet (Test-time Local Converter) wird vor der ersten Epoche entfernt und
nach Ende des Trainings wieder angehängt. Der Backward-Pass durch einen
lokalen Pool mit festem Fenster würde nicht der Verwendung des Checkpoints bei
der Inferenz entsprechen.

Unter [Training](/docs/train) findest du Datensätze, Datenaugmentierung,
Multi-GPU und Logger.

## Validierung

`val()` gibt ein Dictionary mit `metrics/PSNR` und `metrics/SSIM` zurück,
berechnet in RGB über die gesamte gültige Arbeitsfläche. SSIM verwendet ein
11x11-Gauß-Fenster mit Sigma 1.5. `fitness` für die Auswahl des besten
Checkpoints ist der PSNR-Wert. `data` verweist auf dasselbe gepaarte Bildformat
wie beim Training.

<code-tabs name="val" />

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint,
gibt dasselbe `Results`-Objekt zurück und enthält das Ausgabebild in `restored`.
NAFNet wird mit fester räumlicher Auflösung exportiert. `imgsz` muss durch den
Downsampling-Faktor des Netzwerks teilbar sein, bei beiden Architekturbreiten
16. Wenn `dynamic=True` gesetzt ist, bleibt nur die Batch-Dimension dynamisch.
Höhe und Breite werden beim Export festgelegt.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
