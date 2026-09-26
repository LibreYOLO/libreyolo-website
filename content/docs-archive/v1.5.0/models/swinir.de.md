---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: 4-fache Bild-Super-Resolution in LibreYOLO'
description: >-
  Verwende SwinIR in LibreYOLO für eine 4-fache Bild-Super-Resolution.
  Installiere, nutze, validiere und exportiere die Checkpoints in Lightweight,
  Medium und Large.
lead: >-
  Ein Swin-Transformer-Netzwerk zur Bildrestauration. LibreYOLO bietet Inferenz
  und Validierung für seine Checkpoints mit 4-facher Super-Resolution: den
  offiziellen Lightweight-Generator sowie die Real-World-Generatoren in Medium
  und Large.
keywords:
  - SwinIR
  - Swin Transformer
  - Bild-Super-Resolution
  - Bildrestauration
  - Residual Swin Transformer Block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Kachelverarbeitung für große Bilder
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRl-restore.pt")


        # tile teilt den Vorwärtsdurchlauf in überlappende Kacheln und fügt

        # die Übergänge wieder zusammen; tile_pad ist der Rand um jede Kachel,

        # der vor dem Zurückschneiden hinzugefügt wird. Beides sind nur in
        Python

        # verfügbare Schlüsselwortargumente, keine CLI-Flags.

        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")

        # Wenn imgsz fehlt, wird standardmäßig eine kleine interne Patch-Größe
        # und nicht deine Arbeitsauflösung verwendet. Übergib deshalb die Größe,
        # die dein Deployment tatsächlich an das Modell übergibt.
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Exportierte Datei verwenden
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Die Factory leitet anhand der Dateiendung weiter, daher wird ein
        exportiertes Artefakt

        # wie jeder Checkpoint geladen und gibt dasselbe Results-Objekt zurück.

        model = LibreYOLO("LibreSwinIRm-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Installation

SwinIR benötigt kein optionales Extra. Alle Importe sind in der Basisinstallation enthalten.

```bash
pip install libreyolo
```

## Vorhersage

Die Gewichte werden bei der ersten Verwendung von Hugging Face heruntergeladen und lokal zwischengespeichert.

<code-tabs name="predict" />

Ein Restaurationsergebnis enthält keine Boxen. `result.restored` ist ein dichtes RGB-Bild vom Typ uint8 mit der Form `(H, W, 3)`, dessen Canvas in jeder Dimension viermal so groß wie die Eingabe ist. `save=True` schreibt dieses Bild direkt statt einer annotierten Darstellung. Die Eingabe wird auf ein Vielfaches von 8 aufgefüllt und nicht skaliert, sodass die Vorhersage in der ursprünglichen Auflösung des Fotos ausgeführt wird. Eine Quelle, die größer als der verfügbare Speicher ist, lässt sich mit `tile` und `tile_pad` aufteilen. Dabei werden die Übergänge zwischen den Kacheln in der Ausgabe zusammengefügt. Unter [Vorhersage](/docs/predict) findest du Informationen zu Quellen, Streaming und Ergebnisverarbeitung.

## Varianten

Es gibt drei Größen, alle mit fester 4-facher Hochskalierung. `s` ist der offizielle Lightweight-Generator mit vier Stufen aus Residual Swin Transformer Blocks (RSTB) und Pixel-Shuffle-Direct-Upsampling. `m` und `l` sind die Real-World-Generatoren in Medium und Large. Sie besitzen sechs beziehungsweise neun RSTB-Stufen und einen Upsampler aus Nächster-Nachbar-Skalierung und Faltung, der für reale Bildverschlechterungen statt nur für bikubische Verkleinerung ausgelegt ist.

## Validierung

`val()` misst PSNR und SSIM zwischen der restaurierten Ausgabe und einem unverfälschten Zielbild. Beide Metriken werden in RGB auf dem ursprünglichen Canvas ohne Randbeschnitt oder Skalierung berechnet. SSIM verwendet ein gaußsches Fenster von 11x11 mit Sigma 1,5 und bildet den Mittelwert über die drei Farbkanäle.

<code-tabs name="val" />

Das Datensatzargument ist eine YAML-Datei, die ein Verzeichnis mit beeinträchtigten Eingabebildern einem Verzeichnis mit unverfälschten Zielbildern gleicher Auflösung zuordnet. Die genauen Schlüssel findest du unter [Datensatzformate](/docs/reference/dataset-formats).

## Export

<export-matrix />

Ein exportiertes Artefakt wird über seine Dateiendung wieder durch `LibreYOLO()` geladen. Eine `.onnx`- oder `.engine`-Datei verhält sich daher wie ein Checkpoint und gibt dasselbe `Results`-Objekt zurück. ExecuTorch und alle in der Matrix als gesperrt markierten Formate sind für diese Familie nicht verfügbar. ONNX, TorchScript, TensorRT, OpenVINO und TFLite werden unterstützt. Unter [Export](/docs/export) findest du die Argumente, die jedes Format akzeptiert, sowie zusätzliche Argumente einzelner Formate.

<code-tabs name="export" />

## Checkpoints

Alle für diese Familie veröffentlichten Gewichtsdateien.

<checkpoint-table />

## Lizenzierung

<provenance-box></provenance-box>

## Zitieren

<citation-block />
