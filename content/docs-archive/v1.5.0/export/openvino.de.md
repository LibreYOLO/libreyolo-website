---
title: OpenVINO
seo_title: Export nach OpenVINO IR aus LibreYOLO
description: >-
  Konvertiere ein LibreYOLO-Modell in die OpenVINO IR: das Paar aus model.xml
  und model.bin, FP16-Kompression der Gewichte, NNCF-INT8 und Inferenz auf CPU,
  GPU oder NPU.
lead: >-
  Die OpenVINO IR ist Intels Runtime-Format: ein model.xml-Graph neben einem
  model.bin-Gewichts-Blob. LibreYOLO exportiert eine ONNX-Zwischenstufe,
  konvertiert sie mit ov.convert_model und schreibt eine metadata.yaml in
  dasselbe Verzeichnis.
keywords:
  - yolo nach openvino exportieren
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - nncf int8 quantisierung
  - openvino npu
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="openvino")
    mono: true
  - label: Schreibt
    value: 'Ein Verzeichnis mit model.xml, model.bin und metadata.yaml'
  - label: Extra
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Zurückladen
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: Formen
    value: 'Folgt der ONNX-Zwischenstufe: dynamischer Batch bei dynamic=True'
  - label: Präzision
    value: >-
      FP32, FP16-Kompression der Gewichte (half=True), INT8 über NNCF (int8=True
      mit data=)
verification: >-
  Gelesen aus libreyolo/export/openvino.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/openvino.py und pyproject.toml
  im dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # Die IR entsteht aus einer ONNX-Zwischenstufe, daher beide Extras.
        pip install "libreyolo[onnx,openvino]"
    - label: INT8 braucht zusätzlich NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt das Verzeichnis weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Argumente
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True behält eine dynamische Batch-Achse in der IR
            half=False,       # True speichert FP16-Gewichte
            int8=False,       # True führt NNCF-Quantisierung nach dem Training aus
            data=None,        # nötig, wenn int8=True
            output_path=None, # None schreibt weights/<stem>_openvino
        )
  int8:
    - label: INT8 mit Kalibrierungsdaten
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # nötig: für dieses Format gibt es keinen Standard
            fraction=1.0,
        )
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Das Gerät auswählen
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" und "cpu" gehen auf CPU, "gpu" und "cuda" auf GPU,
        # alles andere wird großgeschrieben durchgereicht, etwa "npu" -> NPU.
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: Reines OpenVINO
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # Klassennamen, Aufgabe und Eingabegröße stehen in metadata.yaml neben
        der IR.

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Preprocessing und Postprocessing liegen auf diesem Weg bei dir.
  support:
    - label: Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## Installation

<code-tabs name="install" />

Die Konvertierung läuft über eine ONNX-Zwischenstufe, deshalb gehört das Extra
`onnx` zur Voraussetzung und ist keine optionale Beigabe. NNCF wird separat
installiert und nur für `int8=True` gebraucht.

## Export

<code-tabs name="export" />

Das Artefakt ist ein Verzeichnis, keine Datei. `weights/LibreYOLO9t_openvino`
enthält `model.xml`, `model.bin` und `metadata.yaml`, und bei `half=True` wird
`_fp16` vor dem Suffix eingefügt. Verschiebe oder kopiere das ganze Verzeichnis;
die drei Dateien sind ein Artefakt.

`half=True` setzt beim Speichern `compress_to_fp16`. Das ist eine Kompression der
Gewichte in der IR, keine Änderung der Inferenzpräzision, die das Gerät zur
Laufzeit wählt.

### INT8

<code-tabs name="int8" />

`int8=True` führt die NNCF-Quantisierung nach dem Training über einen
LibreYOLO-Kalibrierungsloader mit dem Preset mixed aus, und `data` ist Pflicht:
für dieses Format gibt es keinen Rückfall auf acht Bilder. Fehlt NNCF, kommt ein
`ImportError`, der den Installationsbefehl nennt.

## Ausführung des Artefakts

<code-tabs name="run" />

`LibreYOLO()` erkennt jedes Verzeichnis, das `model.xml` enthält, und liefert
dasselbe `Results`-Objekt wie der Checkpoint; Klassennamen, Aufgabe, Eingabegröße
und Pose-Schema liest es aus `metadata.yaml`.

Der Gerätestring wird abgebildet und nicht direkt durchgereicht. `auto` und `cpu`
kompilieren beide für CPU, `gpu` und `cuda` beide für GPU, und jeder andere Wert
wird großgeschrieben an OpenVINO übergeben. So erreichst du ein NPU-Ziel.

Das dritte Snippet richtet sich an Leser ohne installiertes LibreYOLO.
Preprocessing, Decoding, NMS und die Umrechnung der Koordinaten liegen dort bei
dir, und die Klassennamen stehen nur in `metadata.yaml`.

## Einschränkungen

Eine IR ohne ihre `metadata.yaml` lädt weiterhin, das Backend fällt dann aber auf
80 Klassen und die Aufgabe Objekterkennung zurück, was für alles andere falsch
ist. Lass das Verzeichnis vollständig.

Vor dem Tracing blockiert: YOLO9-Segmentierung, RTMDet-Ins-Segmentierung,
Objekterkennung mit SSD, Faster R-CNN und RetinaNet sowie Matting mit BiRefNet
oder FeyNobg, wo OpenVINO 2026.2 die standardmäßige ONNX-Operation
`DeformConv-19` des gemeinsamen Matte-Decoders nicht absenken kann.

Wo eine Kombination weder validiert noch blockiert ist, steht der Konverterpfad
zur Verfügung und das Projekt hat dafür keine OpenVINO-Runtime-Parität erfasst.
Mehrere Kombinationen sind mit einem ausdrücklichen Kontext validiert, zum
Beispiel die semantische Segmentierung mit DeepLabV3 bei fester Eingabe von
520 mal 520 auf OpenVINO 2026.2 mit der Standard-Inferenzpräzision der CPU, und
L2CS-Blickschätzung bei festem Gesichtsausschnitt von 448 mal 448.
`libreyolo formats` gibt diesen Kontext pro Kombination aus.

Das vollständige Raster aus Familien und Aufgaben zeigt
[die Export-Matrix](/docs/reference/export-matrix). Für eine einzelne
Kombination:

<code-tabs name="support" />
