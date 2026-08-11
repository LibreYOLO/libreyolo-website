---
title: TFLite
seo_title: Export nach TFLite (LiteRT) aus LibreYOLO
description: >-
  Exportiere ein LibreYOLO-Modell mit onnx2tf in einen .tflite-FlatBuffer:
  statische Shapes, nur FP32, NHWC-Eingaben und die Familien, die sauber
  konvertieren.
lead: >-
  TFLite ist das FlatBuffer-Format, das LiteRT auf mobilen und eingebetteten
  Zielen ausführt. LibreYOLO exportiert einen statischen ONNX-Graphen,
  konvertiert ihn mit onnx2tf im Modus flatbuffer_direct und schreibt die
  Modell-Metadaten als JSON-Sidecar-Datei neben das Artefakt.
keywords:
  - yolo nach tflite exportieren
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - nhwc eingabe tflite
  - inferenz auf edge-geräten
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tflite")
    mono: true
  - label: Schreibt
    value: Eine .tflite-Datei plus eine .tflite.json-Metadaten-Sidecar-Datei
  - label: Extra
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Lädt zurück
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: Shapes
    value: Nur statisch. dynamic=True wird abgelehnt.
  - label: Präzision
    value: Nur FP32. half=True und int8=True werden abgelehnt.
  - label: Voraussetzung
    value: >-
      Python 3.12 oder neuer, weil onnx2tf 2.4.x keine älteren Wheels
      veröffentlicht
verification: >-
  Gelesen aus libreyolo/export/tflite.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tflite.py und pyproject.toml
  auf dem dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # LiteRT ist Googles aktueller Name für TensorFlow Lite. Beide Extras
        # installieren dieselbe Toolchain und liefern dieselbe .tflite-Ausgabe.
        pip install "libreyolo[tflite]"
    - label: Zuerst die Python-Version prüfen
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Schreibt weights/LibreYOLO9t.tflite und
        weights/LibreYOLO9t.tflite.json

        path = model.export(format="tflite", imgsz=640)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" wird als Alias akzeptiert und nutzt denselben Exporter.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Argumente
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int, oder (Höhe, Breite)
            batch=1,
            simplify=True,    # onnxsim über dem ONNX-Zwischenstand
            output_path=None, # None schreibt weights/<stem>.tflite
            verbose=False,    # True gibt das onnx2tf-Log aus
        )

        # dynamic=True löst ValueError aus: der Konverter braucht feste Shapes.
        # half=True und int8=True werden vor dem Tracing abgelehnt.
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Reines LiteRT
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, nicht NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Klassennamen, Aufgabe und Eingabegröße stehen in der Sidecar-Datei.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Preprocessing, die NCHW-zu-NHWC-Transposition und Postprocessing: bei
        dir.
  support:
    - label: Eine Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## Installation

<code-tabs name="install" />

Das Extra zieht `onnx2tf` für die Konvertierung und `ai-edge-litert` für das
Ausführen des Ergebnisses, beide hinter einem Marker für Python 3.12. Auf einem
älteren Interpreter löst der Export einen `ImportError` aus, der die
Versionsanforderung benennt, statt im Konverter zu scheitern.

`libreyolo[litert]` installiert exakt dasselbe. Der Format-String `litert` ist ein
Alias für `tflite`, und die Ausgabedatei ist so oder so eine `.tflite`.

## Export

<code-tabs name="export" />

Familie und Aufgabe werden geprüft, bevor sonst irgendetwas passiert, eine nicht
unterstützte Kombination scheitert also sofort mit dem konkreten Konverter- oder
Runtime-Fehler, der sie ausgeschlossen hat, und nicht mit einer allgemeinen
Meldung. Die Konvertierung selbst ist ein Subprozess-Aufruf von `onnx2tf` im Modus
`flatbuffer_direct` über einem statischen ONNX-Zwischenstand.

Die Metadaten liegen in einer Sidecar-Datei. `weights/LibreYOLO9t.tflite.json`
trägt Familie, Aufgabe, Klassennamen, Eingabegröße und Pose-Schema; der FlatBuffer
selbst hat kein LibreYOLO-Metadatenfeld, die beiden Dateien gehören also zusammen.

## Ausführen des Artefakts

<code-tabs name="run" />

`LibreYOLO()` entscheidet anhand der Endung `.tflite` und liefert dasselbe
`Results`-Objekt wie der Checkpoint. Das Backend liest die Sidecar-Datei,
transponiert den NCHW-Blob nach NHWC, wenn der Interpreter eine
Channels-last-Eingabe verlangt, wendet Quantisierungs-Scale und Zero Point des
Interpreters an, wo vorhanden, und transponiert die Ausgaben zurück in das Layout,
das LibreYOLOs Postprocessing erwartet.

Das zweite Snippet ist der Weg über die reine Runtime. Preprocessing, die
Layout-Transposition, Decoding, NMS und die Skalierung der Koordinaten liegen dort
alle bei dir, und das Layout-Detail wird am ehesten übersehen: onnx2tf erzeugt
Channels-last-Eingaben, ein Blob der Form `(1, 3, 640, 640)` bindet also nicht.

## Einschränkungen

Nur statische Shapes. `dynamic=True` löst vor dem Tracing einen `ValueError` aus,
und das Export-Canvas ist auf den Wert festgelegt, zu dem `imgsz` aufgelöst wurde.

Nur FP32. `half=True` und `int8=True` werden beide bei der Validierung abgelehnt,
ein quantisiertes Deployment ist über diesen Exporter heute also nicht erreichbar.

Die Abdeckung ist hier schmaler als bei den Graph-Formaten, und sie entscheidet
sich über Messungen statt über die Familie. Zu den validierten Kombinationen
gehören die Erkennung mit YOLO9, YOLOX und YOLO-NAS, die semantische Segmentierung
mit PIDNet, die vier CNN-Klassifikationsfamilien, Embedding mit DINOv2 und
SigLIP2, Klassifikation mit SigLIP2, Kantenerkennung mit TEED und DexiNed sowie
Restaurierung mit Real-ESRGAN und SwinIR. Bei SwinIR kommt eine Einschränkung
dazu: Die Parität hält, wenn die Quellabmessungen exakt zum Export-Canvas passen,
und kleinere Quellen werden vor dem Lauf des Transformers auf das Canvas gepaddet,
was von nativer Inferenz mit variabler Größe abweichen kann.

Die blockierten Einträge benennen den konkreten Fehler, und das ist es wert,
gelesen zu werden, bevor du einen Workaround versuchst. Ein paar Beispiele: Die
RF-DETR-Erkennung konvertiert bei ihrem nativen 384er-Canvas, aber LiteRT kann sie
nicht allozieren, weil `STRIDED_SLICE` eine Eingabe oberhalb des unterstützten
5-D-Rangs bekommt; PicoDet wird abgelehnt, weil ein `RESHAPE` 19.200
Eingabeelemente auf 9.600 Ausgabeelemente abbildet; D-FINE bringt den Konverter
bei der Shape-Behandlung von `GatherElements` zum Absturz; RTMDet exportiert und
lädt mit intakter Roh-Parität zurück, die öffentlichen Boxen fallen aber auf
0.911 IoU bei 29.9 px Koordinatendrift.

Das vollständige Raster aus Familien und Aufgaben steht in
[der Export-Matrix](/docs/reference/export-matrix). Für eine einzelne Kombination,
samt dem Grund-String hinter einer Blockade:

<code-tabs name="support" />
