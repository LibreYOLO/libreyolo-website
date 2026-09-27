---
title: MNN
seo_title: Export nach MNN aus LibreYOLO
description: >-
  Exportiere einen LibreYOLO-Detektor über ONNX und mnnconvert nach MNN: eine
  feste NCHW-Form, FP32 auf der CPU und ein Metadaten-Sidecar, das der
  Runtime-Vertrag verlangt.
lead: >-
  MNN ist Alibabas leichtgewichtige Inferenz-Engine. LibreYOLO exportiert einen
  statischen ONNX-Graphen, konvertiert ihn mit dem Werkzeug mnnconvert aus dem
  MNN-Paket und schreibt ein JSON-Sidecar, das die Eingabe- und Ausgabenamen,
  die feste Eingabeform und die Klassennamen festhält.
keywords:
  - yolo nach mnn exportieren
  - mnnconvert
  - mnn inferenz
  - objekterkennung auf mobilgeräten
  - feste nchw form
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="mnn")
    mono: true
  - label: Schreibt
    value: Eine .mnn-Datei plus ein .mnn.json-Metadaten-Sidecar
  - label: Extra
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Zurückladen
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: Formen
    value: Feste NCHW-Form. dynamic=True wird abgelehnt.
  - label: Präzision
    value: 'Nur FP32, nur CPU.'
  - label: Aufgaben
    value: Nur Objekterkennung in dieser Version
verification: >-
  Gelesen aus libreyolo/export/mnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/mnn.py und pyproject.toml im
  dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # Das Extra enthält libreyolo[onnx]: MNN konvertiert über ONNX.
        pip install "libreyolo[mnn]"
    - label: 'Prüfen, ob der Konverter im PATH liegt'
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt weights/LibreYOLO9t.mnn und weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Argumente
      language: python
      code: >
        model.export(
            format="mnn",
            imgsz=640,        # int oder (Höhe, Breite)
            batch=1,          # fest ins Artefakt geschrieben
            simplify=True,    # onnxsim über das ONNX-Zwischenformat
            output_path=None, # None schreibt weights/<stem>.mnn
            verbose=False,    # True streamt das mnnconvert-Log
        )


        # dynamic=True wirft ValueError. half=True und int8=True werden
        abgelehnt.
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Reines MNN
      language: python
      code: >
        import json


        import MNN

        import numpy as np


        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))

        print(meta["mnn_input_names"], meta["mnn_output_names"],
        meta["mnn_input_shape"])


        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )

        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )


        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)

        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )

        outputs = module.forward([input_var])

        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # Preprocessing und Postprocessing liegen auf diesem Weg bei dir.
  support:
    - label: Eine Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## Installation

<code-tabs name="install" />

Das Extra enthält `libreyolo[onnx]`, weil die Konvertierung über ein
ONNX-Zwischenformat läuft. Es bringt außerdem die ausführbare Datei `mnnconvert`
mit, nach der der Exporter zuerst neben dem aktiven Python-Interpreter und danach
in `PATH` sucht. Fehlt der Konverter, wirft er einen `ImportError`, der den
Installationsbefehl nennt, statt mitten in der Konvertierung zu scheitern.

## Export

<code-tabs name="export" />

Bevor er den Graphen weiterreicht, liest der Exporter den ONNX-Eingabevertrag und
lehnt alles ab, was er nicht ausdrücken kann: mehr als eine Bildeingabe oder eine
Eingabeform mit einer symbolischen Dimension. MNN verlangt in dieser Version eine
vollständig feste NCHW-Form, und `batch` wird fest ins Artefakt geschrieben statt
beim Laden ausgehandelt.

Das Sidecar ist keine optionale Buchhaltung. `weights/LibreYOLO9t.mnn.json` hält
die Eingabe- und Ausgabenamen fest, die feste Eingabeform, den Batch, die
Klassennamen, die verwendete MNN-Version und das Backend, für das das Artefakt
gebaut wurde, und die Runtime prüft jedes einzelne dieser Felder beim Laden.

Unter Windows schließt MNN 3.6.1 die Konvertierung manchmal ab und bricht dann
beim Abbau des Prozesses mit einer Zugriffsverletzung oder einem Fail-Fast-Status
ab. Der Exporter erkennt genau diese Exit-Codes und wertet die Konvertierung als
erfolgreich, wenn die Ausgabedatei vorhanden ist.

## Ausführung des Artefakts

<code-tabs name="run" />

`LibreYOLO()` entscheidet anhand der Endung `.mnn` und liefert dasselbe
`Results`-Objekt wie der Checkpoint. Das Laden ist mit Absicht streng: Das Sidecar
muss `format=mnn`, `mnn_backend=cpu`, `dynamic=false`, `precision=fp32`, eine
Größe, eine Erkennungsaufgabe, eine feste positive NCHW-Form, die zur notierten
Bildgröße passt, und Klassennamen für jeden Index von 0 bis `nc - 1` deklarieren.
Jede Abweichung führt zu einem Fehler statt zu einer Vermutung.

Eine Vorhersage mit einem anderen `imgsz`, als das Artefakt gebaut wurde, führt
ebenfalls zu einem Fehler, und `device` wird mit einer Warnung ignoriert, weil
MNN-Exporte hier auf der CPU laufen.

Das zweite Snippet ist der Weg über die reine Runtime. Preprocessing, Decoding,
NMS und die Umrechnung der Koordinaten liegen dort bei dir, und die Eingabe- und
Ausgabenamen kommen aus dem Sidecar, weil der Modul-Loader von MNN sie explizit
haben will.

## Einschränkungen

Nur Objekterkennung. Das Backend lehnt jede andere Aufgabe beim Laden ab, und die
Exportseite passt dazu: Außerhalb der notierten Kombinationen bricht der Preflight
mit „MNN v1 has no implemented runtime contract for this family and task.“ ab.

FP32, CPU, feste Form. `dynamic=True` wirft einen `ValueError`, und `half=True`
sowie `int8=True` werden bei der Validierung abgelehnt.

Validierte Erkennungsfamilien sind YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM und YOLO-NAS, jeweils abgedeckt durch
Konvertierung, ein frisches Neuladen des Artefakts, Ausführung auf der MNN-CPU,
Metadaten-Prüfungen und übereinstimmende Detektionsparität nach NMS gegenüber dem
PyTorch-Modell. DEIMv2 konvertiert, lädt neu, führt aus und erhält die Detektionen
nach NMS, aber die ONNX-Zwischenroute hat keine vollständige Score-Parität auf
Query-Ebene, deshalb ist die Familie als verfügbar statt als validiert notiert.

Das vollständige Raster aus Familien und Aufgaben zeigt
[die Export-Matrix](/docs/reference/export-matrix). Für eine einzelne Kombination:

<code-tabs name="support" />
