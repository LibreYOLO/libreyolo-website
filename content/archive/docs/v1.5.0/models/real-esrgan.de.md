---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: Bild-Super-Resolution in LibreYOLO'
description: >-
  Nutze Real-ESRGAN in LibreYOLO für praktische Bild-Super-Resolution mit 4x, 2x
  und einem schnellen 4x-Tier. Installiere, sage vorher, validiere und
  exportiere.
lead: >-
  Ein praktischer blinder Super-Resolution-Upscaler, der mit synthetischen
  Beeinträchtigungen statt ausschließlich bikubischem Downscaling trainiert
  wurde. LibreYOLO bietet Inferenz und Validierung für seine 4x-, 2x- und
  schnellen 4x-Checkpoints.
keywords:
  - real-esrgan python
  - rrdbnet
  - srvggnetcompact
  - bild hochskalieren
  - bildwiederherstellung
  - blind super-resolution
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Gekachelt für große Bilder
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # tile teilt den Vorwärtslauf in überlappende Kacheln und verblendet die

        # Nähte wieder. tile_pad ist der Rand um jede Kachel vor dem
        Zuschneiden.

        # Beide sind reine Python-Schlüsselwortargumente und keine CLI-Flags.

        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # Ohne Angabe nutzt imgsz eine kleine interne Patch-Größe statt deiner

        # Arbeitsauflösung. Übergib die tatsächliche Eingabegröße des
        Deployments.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory entscheidet anhand der Dateiendung, daher wird ein
        Exportartefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Installation

Real-ESRGAN benötigt kein optionales Zusatzpaket. Alle Importe sind in der
Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Ein Wiederherstellungsergebnis enthält keine Boxen. `result.restored` ist ein
dichtes uint8-RGB-Bild der Form `(H, W, 3)` auf einer Arbeitsfläche, die in
jeder Dimension `Results.restore_scale` mal größer als die Eingabe ist.
`save=True` schreibt dieses Bild direkt statt einer annotierten Darstellung.
Die Eingabe wird in RGB umgewandelt, und ein Alphakanal wird entfernt. Wenn
eine Quelle größer als der verfügbare Speicher ist, kannst du sie mit `tile`
und `tile_pad` aufteilen. Die Nähte der Kacheln werden in der Ausgabe
verblendet. Unter [Vorhersage](/docs/predict) findest du Quellen, Streaming und
die Verarbeitung von Ergebnissen.

## Varianten

Es gibt drei Checkpoints, benannt nach ihrem Skalierungsfaktor. `x4` ist
RRDBNet (`RealESRGAN_x4plus`) mit 23 Residual-in-Residual-Dense-Blöcken und der
Qualitätsstandard für 4x. `x2` verwendet dieselbe RRDBNet-Architektur für 2x.
`x4t` ist SRVGGNetCompact (`realesr-general-x4v3`), ein kleinerer und
schnellerer Generator für Video und geringere Latenz bei 4x. Das allgemeine
Upstream-Modell liefert außerdem ein verbundenes Netzwerk für die
Entrauschungsstärke, das bei der Inferenz eingemischt wird. Diese Stärke ist
nicht Bestandteil der Portierung, die den Basisgenerator `x4t` ausführt.

## Validierung

`val()` misst PSNR und SSIM zwischen der wiederhergestellten Ausgabe und einem
sauberen Zielbild. Beide werden in RGB auf der ursprünglichen Arbeitsfläche
ohne Randbeschnitt oder Größenänderung berechnet. SSIM verwendet ein
11x11-Gauß-Fenster mit Sigma 1.5 und mittelt über die drei Farbkanäle.

<code-tabs name="val" />

Das Datensatzargument ist eine YAML, die ein Verzeichnis beeinträchtigter
Eingabebilder mit einem Verzeichnis sauberer Zielbilder gleicher Auflösung
paart. Unter [Datensatzformate](/docs/reference/dataset-formats) findest du die
genauen Schlüssel.

## Export

<export-matrix />

Ein exportiertes Artefakt wird anhand seiner Dateiendung wieder über `LibreYOLO()`
geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint
und gibt dasselbe `Results`-Objekt zurück. [Export](/docs/export) führt die
Argumente auf, die jedes Format akzeptiert, sowie die Zusatzpakete, die einige
davon benötigen.

<code-tabs name="export" />

## Checkpoints

Alle veröffentlichten Gewichtsdateien dieser Familie.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
