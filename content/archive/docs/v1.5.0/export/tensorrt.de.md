---
title: TensorRT
seo_title: Export nach TensorRT aus LibreYOLO
description: >-
  Eine TensorRT-Engine aus einem LibreYOLO-Modell bauen: das
  ONNX-Zwischenformat, FP16- und INT8-Builds, dynamische Batch-Profile und die
  Grenzen der Engine-Portabilität.
lead: >-
  TensorRT kompiliert einen Graphen in eine Engine, die auf genau eine GPU
  abgestimmt ist. LibreYOLO exportiert zuerst ein ONNX-Zwischenformat, parst es
  mit dem ONNX-Parser von TensorRT, baut die Engine und schreibt die
  Modell-Metadaten als JSON-Sidecar-Datei daneben.
keywords:
  - yolo nach tensorrt exportieren
  - tensorrt engine
  - trt fp16
  - tensorrt int8 kalibrierung
  - tensorrt optimierungsprofil
  - dynamischer batch tensorrt
  - hardware compatibility level
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tensorrt")
    mono: true
  - label: Schreibt
    value: Eine .engine-Datei plus eine .engine.json-Metadaten-Sidecar-Datei
  - label: Extra
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Lädt zurück
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: Shapes
    value: >-
      Standardmäßig statisch; dynamic=True fügt ein Optimierungsprofil für die
      Batch-Achse hinzu
  - label: Präzision
    value: 'FP32, FP16 (half=True), INT8 (int8=True mit data=)'
  - label: Voraussetzungen
    value: >-
      Eine NVIDIA-GPU beim Build und zur Laufzeit. Engines wechseln nicht
      zwischen GPU-Architekturen.
verification: >-
  Gelesen aus libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py und pyproject.toml
  auf dem dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # Die Engine entsteht aus einem ONNX-Zwischenformat, daher beide Extras.
        pip install "libreyolo[onnx,tensorrt]"
    - label: Die Toolchain vor dem Build prüfen
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Schreibt weights/LibreYOLO9t_fp16.engine und
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Argumente
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # nötig bei int8=True
            dynamic=False,
            workspace=4.0,                  # GiB Scratch beim Build
            min_batch=1,                    # Grenzen des dynamischen Profils
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # oder "ampere_plus"
            gpu_device=0,                   # Build-Gerät auf Multi-GPU-Host
            verbose=False,
        )
  dynamic:
    - label: Engine mit dynamischem Batch
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Das ONNX-Zwischenformat braucht die dynamische Batch-Achse, damit
        # das Profil überhaupt etwas zum Binden hat.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 mit Kalibrierungsdaten
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # Pflicht: für dieses Format gibt es keinen Default
            fraction=1.0,
        )
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Reines TensorRT
      language: python
      code: |
        import json

        import tensorrt as trt

        path = "weights/LibreYOLO9t_fp16.engine"
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Klassennamen, Aufgabe und Eingabegröße stehen in der Sidecar-Datei.
        # Buffer-Allokation, Pre- und Postprocessing liegen hier bei dir.
        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Eine Familie und Aufgabe vor dem Build prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## Installation

Sowohl der Build als auch die Ausführung brauchen eine NVIDIA-GPU mit
funktionierendem CUDA-Stack. Für dieses Format gibt es keinen CPU-Fallback.

<code-tabs name="install" />

Das Extra `tensorrt` pinnt `tensorrt-cu12` und `pycuda`, und der Marker lässt beide
unter macOS weg. Auf einem Jetson nimm dieses Extra nicht: Es pinnt einen
CUDA-12-Build gegen eine CUDA-13-Plattform. Nutze stattdessen das TensorRT, das
JetPack installiert, wie unter [NVIDIA Jetson](/docs/export/jetson) beschrieben.

## Export

<code-tabs name="export" />

Der Export läuft in zwei Schritten. Schritt eins schreibt ein ONNX-Zwischenformat an
einen temporären Pfad, Schritt zwei parst es und baut die Engine, und das
Zwischenformat wird danach entfernt. `workspace` ist Scratch-Speicher zur Build-Zeit
in GiB; ein größerer Wert lässt den Builder mehr Kernel ausprobieren und wirkt sich
nicht auf den Speicher zur Inferenzzeit aus.

Die Metadaten-Sidecar-Datei wird als `<engine>.json` neben die Engine geschrieben
und hält die Präzision fest, die der Build tatsächlich erreicht hat. Fehlt der GPU
schnelles FP16 oder schnelles INT8, warnt der Builder und fällt zurück, und die
Sidecar-Datei meldet die Präzision, die herauskam, statt der angeforderten.

Unter FP16 wird ein ViT-Backbone im Graphen erkannt und seine Float-Schichten werden
auf FP32 festgesetzt. Backbones im DINOv2-Stil laufen in FP16 über und erzeugen NaN,
deshalb setzt der Build `OBEY_PRECISION_CONSTRAINTS` und meldet
`FP16 (FP32 ViT backbone)`. Bei CNN-Backbones bewirkt der Durchlauf nichts.

### Dynamischer Batch

<code-tabs name="dynamic" />

`dynamic=True` fügt ein Optimierungsprofil von `min_batch` bis `max_batch` hinzu,
optimiert bei `opt_batch`, und hält diese drei Werte in der Sidecar-Datei fest. Das
Profil wird nur hinzugefügt, wenn das ONNX-Zwischenformat tatsächlich eine
dynamische Batch-Dimension trägt; sonst protokolliert der Build, dass er statisch
optimiert, und läuft weiter.

### INT8

<code-tabs name="int8" />

INT8 nutzt den Entropie-Kalibrator von TensorRT über einen
LibreYOLO-Kalibrierungsloader, und `data` ist Pflicht: Dieses Format hat keinen
Fallback auf acht Bilder. Die Kalibrierung braucht `cuda-python` oder `pycuda` für
den Device-Buffer. Der Kalibrierungs-Cache wird über einen Hash der ONNX-Bytes
geschlüsselt, damit die Skalen eines Modells nie für ein anderes wiederverwendet
werden, das zufällig auf denselben Ausgabepfad schreibt.

`half=True` und `int8=True` zusammen erzeugen eine Warnung und bauen INT8, das einen
FP16-Fallback für Schichten behält, die TensorRT nicht quantisieren kann.

## Ausführung des Artefakts

<code-tabs name="run" />

`LibreYOLO()` entscheidet anhand der Endung `.engine`, liest Klassennamen, Aufgabe
und Pose-Schema aus der Sidecar-Datei und liefert dasselbe `Results`-Objekt wie der
Checkpoint. Ohne vorhandenes CUDA-Gerät löst es sofort einen Fehler aus.

Das zweite Snippet ist der Weg über die reine Runtime. Buffer-Allokation auf Host
und Device, Preprocessing, Decoding, NMS und die Umrechnung der Koordinaten liegen
dort vollständig bei dir, und die Engine selbst trägt keine Klassennamen, die
Sidecar-Datei muss sie also begleiten.

## Einschränkungen

Eine serialisierte Engine ist an die GPU-Architektur, den Treiber-Stack und die
TensorRT-Version gebunden, die sie gebaut hat. Eine auf einer Workstation gebaute
Engine lädt auf einer anderen Architektur nicht, deshalb läuft der Build-Schritt auf
der Zielmaschine. `hardware_compatibility="ampere_plus"` tauscht etwas Performance
gegen Portabilität über Ampere und neuer. Der Wert `"same_compute_capability"` wird
auf `NONE` abgebildet und warnt: Die Engine ist nur für die aktuelle GPU optimiert,
und der Export sagt das, statt eine Portabilität zu behaupten, die er nicht
angewendet hat.

Profiliert wird nur die Batch-Achse. Ein Build mit dynamischen räumlichen
Dimensionen gehört nicht zu diesem Vertrag, deshalb ist FCOS blockiert: Es braucht
dynamisch aufgefüllte Höhe und Breite, um seine Transformation des
Seitenverhältnisses auf 800 mal 1333 zu erhalten.

Vor dem Tracing blockiert: YOLO9-Segmentierung, RTMDet-Ins-Segmentierung, SSD-,
Faster-R-CNN- und RetinaNet-Erkennung sowie BiRefNet- oder FeyNobg-Matting, wo
TensorRT 10.16 auf den gemeinsamen ONNX-Knoten `DeformConv` trifft und ihn nicht
parsen kann, weil `ModulatedDeformConv2d` in der Plugin-Registry fehlt.

Wo eine Kombination weder validiert noch blockiert ist, steht der Konverter-Pfad zur
Verfügung und das Projekt hat dafür keine TensorRT-Runtime-Parität erfasst. Das ist
eine Aussage über die Belege, nicht darüber, ob der Build gelingt.

Das vollständige Raster aus Familien und Aufgaben zeigt
[die Export-Matrix](/docs/reference/export-matrix). Für eine einzelne Kombination:

<code-tabs name="support" />
