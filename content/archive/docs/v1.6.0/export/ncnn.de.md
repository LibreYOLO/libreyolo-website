---
title: ncnn
seo_title: Export nach ncnn aus LibreYOLO
description: >-
  Ein LibreYOLO-Modell über PNNX nach ncnn exportieren: das Paar aus param und
  bin, die feste Eingabegröße, das Umschreiben von Focus bei YOLOX und welche
  Familien sich konvertieren lassen.
lead: >-
  ncnn ist Tencents CPU-Inferenz-Bibliothek für mobile Ziele. LibreYOLO
  konvertiert über PNNX und schreibt einen Graphen model.ncnn.param neben eine
  Gewichtsdatei model.ncnn.bin und eine metadata.yaml, die Familie, Task und
  Klassennamen trägt.
keywords:
  - yolo nach ncnn exportieren
  - pnnx
  - model.ncnn.param
  - ncnn cpu inferenz mobil
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="ncnn")
    mono: true
  - label: Schreibt
    value: 'Ein Verzeichnis mit model.ncnn.param, model.ncnn.bin und metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Zurückladen
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: Formen
    value: Fest. Die Metadaten vermerken dynamic=False unabhängig vom Flag.
  - label: Präzision
    value: Nur FP32. half=True und int8=True werden abgelehnt.
verification: >-
  Gelesen aus libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py und pyproject.toml im
  dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # pnnx konvertiert, ncnn führt das Ergebnis aus.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt das Verzeichnis weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Argumente
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int, oder (Höhe, Breite)
            batch=1,
            simplify=True,    # gilt nur für den ONNX-Fallback
            opset=None,       # auto; gilt nur für den ONNX-Fallback
            output_path=None, # None schreibt weights/<stem>_ncnn
        )

        # half=True und int8=True werden bei der Validierung abgelehnt.
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Reines ncnn
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn nimmt ein einzelnes CHW-Bild, keinen Batch.
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Preprocessing und Postprocessing liegen hier bei dir.
  support:
    - label: Eine Familie und einen Task vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Installation

<code-tabs name="install" />

Das Extra zieht beide Hälften der Toolchain: `pnnx` erledigt die Konvertierung und
`ncnn` führt das Ergebnis aus. Auf dem primären Pfad geht keines von beiden über ONNX.

## Export

<code-tabs name="export" />

Das Artefakt ist ein Verzeichnis. `weights/LibreYOLO9t_ncnn` enthält
`model.ncnn.param`, `model.ncnn.bin` und `metadata.yaml`; alle drei sind ein
Artefakt und wandern gemeinsam.

Die Konvertierung versucht zuerst PNNX direkt aus PyTorch. Schlägt das fehl,
exportiert sie einen statischen ONNX-Graphen in ein temporäres Verzeichnis und ruft
darauf das Kommandozeilenwerkzeug `pnnx` auf, und der Export wirft erst dann einen
Fehler, wenn beide Pfade scheitern, wobei er beide Fehler meldet. `opset` und
`simplify` wirken sich daher nur auf den Fallback aus.

YOLOX braucht ein Umschreiben, um überhaupt zu konvertieren. Seine Focus-Schicht
nutzt Slicing mit Stride, das PNNX nicht abbilden kann, also tauscht der Export sie
gegen `pixel_unshuffle` und permutiert die Eingangskanäle der folgenden Faltung, um
die andere Kanalreihenfolge auszugleichen. Die Ausgabe ist numerisch identisch, und
die ursprünglichen Gewichte werden nach dem Export wiederhergestellt.

## Das Artefakt ausführen

<code-tabs name="run" />

`LibreYOLO()` erkennt jedes Verzeichnis, das `model.ncnn.param` und
`model.ncnn.bin` enthält, liest `metadata.yaml` und gibt dasselbe `Results`-Objekt
zurück wie der Checkpoint.

Das zweite Snippet ist der Pfad über die reine Runtime, und zwei Details
unterscheiden sich von jedem anderen Format hier. ncnn arbeitet auf einem einzelnen
CHW-Bild statt auf einem Batch, es gibt also keine führende Batch-Achse. Die
Blob-Namen stammen aus der `.param`-Datei; PNNX schreibt per Konvention `in0` und
`out0`, und das Backend parst die Datei, statt sie vorauszusetzen. Preprocessing,
Decoding, NMS und das Zurückskalieren der Koordinaten liegen auf diesem Pfad bei
dir.

## Einschränkungen

FP32 auf einer festen Eingabegröße. `half=True` und `int8=True` werden beide bei
der Validierung abgelehnt, und die exportierten Metadaten vermerken `dynamic=False`,
egal was das Flag gesagt hat, damit kein Backend eine Achse annimmt, die der Graph
nicht hat.

Jede Familie im DETR-Stil wird im Preflight abgelehnt: `detr`, `deformable_detr`,
`dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`,
`rfdetr` und `ec`. Die Meldung ist für alle dieselbe, nämlich dass das Modell
Decoder- oder Sampling-Operationen braucht, die es in ncnn nicht gibt, und sie
verweist stattdessen auf ONNX, OpenVINO, TorchScript oder TensorRT.

Was konvertiert, ist auf der konvolutionalen Seite breit aufgestellt: YOLO9 und
YOLO9-E2E, YOLOX, PicoDet, YOLO-NAS für Erkennung und Pose, die älteren Detektoren
YOLO1, YOLO3, YOLO4 und YOLO7, die vier CNN-Klassifikationsfamilien, die semantische
Segmentierung mit PIDNet, die FOMO-Punkterkennung bei festen 96 mal 96, ZipDepth,
NAFNet und Real-ESRGAN.

Blockierte Einträge nennen den konkreten Fehler. Transformer-Graphen lassen in der
Regel nicht unterstützte `pnnx.Expression`-Knoten zurück, was ein Netz ohne
lauffähigen Eingabe-Blob ergibt, und genau daran scheitern DINOv2, CLIP, SigLIP2 und
SegFormer. BiRefNet braucht die deformable Convolution aus torchvision, die PNNX
nicht abbilden kann. Der konvertierte Graph von YOLO2 beendet die ncnn-Runtime unter
Windows mit einer nativen Integer-Division durch null während der Ausgabe-Extraktion.

Das vollständige Raster aus Familien und Tasks findest du in
[der Export-Matrix](/docs/reference/export-matrix). Für eine einzelne Kombination:

<code-tabs name="support" />
