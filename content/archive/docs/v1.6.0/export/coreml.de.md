---
title: Core ML
seo_title: Export nach Core ML aus LibreYOLO
description: >-
  Exportiere einen LibreYOLO-Detektor in ein Core ML .mlpackage: der
  ImageType-Eingabevertrag, FP16, Compute Units, eingebettetes NMS und die vier
  unterstützten Familien.
lead: >-
  Core ML ist Apples On-Device-Modellformat. LibreYOLO führt das Tracing des
  Detektors hinter einem Preprocessing-Wrapper pro Familie durch, damit der
  konvertierte Graph immer eine kanonische RGB-Bildeingabe bekommt, und schreibt
  dann ein .mlpackage im Format ML Program mit angehängten Modell-Metadaten.
keywords:
  - yolo nach coreml exportieren
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms pipeline
  - yolo auf ios
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreml")
    mono: true
  - label: Schreibt
    value: Ein .mlpackage-Bundle (ein Verzeichnis) im Format ML Program
  - label: Extra
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Zurückladen
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") unter macOS
    mono: true
  - label: Formen
    value: Fest. Die Eingabe ist ein ct.ImageType mit harter Form.
  - label: Präzision
    value: 'FP32, FP16 (half=True). Kein INT8.'
  - label: Familien
    value: 'Nur Objekterkennung, für yolox, yolo9, rtdetr und rfdetr'
verification: >-
  Gelesen aus libreyolo/export/coreml.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/coreml.py und pyproject.toml
  im dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt das Bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Argumente
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True konvertiert mit FLOAT16-Präzision
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None schreibt weights/<stem>.mlpackage
        )

        # dynamic wird akzeptiert, aber die Eingabe ist ein ct.ImageType mit
        # fester Form, und die Metadaten notieren so oder so dynamic=False.
  nms:
    - label: Apples NMS-Schicht einbetten
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Nur Objekterkennung mit YOLOX und YOLO9, Batch 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: 'Über LibreYOLO, unter macOS'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # oder cpu_and_ne für die Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Reines coremltools
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # Die Eingabe ist ein Bild namens "image" in fester Exportgröße.
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # Letterboxing und Postprocessing liegen auf diesem Weg bei dir.
  support:
    - label: Eine Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## Installation

<code-tabs name="install" />

Die Vorhersage braucht macOS. `LibreYOLO()` verweigert ein `.mlpackage` auf jeder
anderen Plattform mit einer Meldung, die die aktuelle nennt, und die Support-Matrix
führt diese Kombinationen als verfügbar, weil die Runtime-Parität einen
macOS-Runner braucht.

## Export

<code-tabs name="export" />

Das Bundle wird unter dem Stamm des Checkpoints nach `weights/` geschrieben, mit
angehängtem `_fp16`, wenn `half=True` gesetzt ist. Ein `.mlpackage` ist ein
Verzeichnis, kopiere also den ganzen Baum.

Für jede Familie erfolgt das Tracing hinter einem Preprocessing-Wrapper, damit der
konvertierte Graph genau eine kanonische Eingabe annimmt: RGB, `scale=1/255`, kein
Bias, deklariert als `ct.ImageType`. Der Wrapper fängt die jeweilige Konvention der
Familie ab, also BGR im Bereich 0 bis 255 bei YOLOX, ImageNet-Mittelwert und
-Standardabweichung bei RF-DETR und Identität bei YOLO9 und RT-DETR. Deshalb
übergibt ein Core-ML-Consumer ein gewöhnliches Bild statt eines
familienspezifischen Tensors.

Die Konvertierung zielt auf ML Program mit iOS 15 als minimalem
Deployment-Target. `compute_units` wird im konvertierten Modell gespeichert und
lässt sich beim Laden des Artefakts erneut überschreiben.

Die Modell-Metadaten landen als Strings in `user_defined_metadata`, und genau dort
liest das Backend Familie, Aufgabe, Klassennamen, Eingabegröße und Pose-Schema.

### Eingebettetes NMS

<code-tabs name="nms" />

`nms=True` verpackt das Modell in eine Core-ML-Pipeline, die in Apples
`NonMaximumSuppression`-Schicht endet. Das Ergebnis hat zwei Ausgaben:
`confidence` mit der Form `N` mal Klassenanzahl und `coordinates` mit der Form `N`
mal 4 als normalisiertes `xywh`.

Das gilt nur für die Objekterkennung mit YOLOX und YOLO9 und setzt Batch 1 voraus.
Die Familien im DETR-Stil werden namentlich abgelehnt, weil Set Prediction ein
Top-k über Queries und Klassen ohne IoU-Schritt bildet und diese Schicht nicht
nutzen kann. Auch `max_det` ist hier nicht verfügbar; wenn die Obergrenze für
Detektionen zählt, nimm stattdessen
[eingebettetes NMS in ONNX](/docs/export/onnx).

## Ausführung des Artefakts

<code-tabs name="run" />

`LibreYOLO()` erkennt ein Verzeichnis mit der Endung `.mlpackage` und liefert
dasselbe `Results`-Objekt wie der Checkpoint. `compute_units` ist das einzige
Argument, das die Factory für dieses Format durchreicht, und es akzeptiert `all`,
`cpu_and_gpu`, `cpu_and_ne` und `cpu_only`. Das Argument `device` wird ignoriert,
weil Core ML stattdessen über Compute Units steuert.

Das zweite Snippet ist der Weg über die reine Runtime. Letterboxing, Decoding, NMS
und die Umrechnung der Koordinaten liegen dort bei dir, und die Klassennamen
stehen in `user_defined_metadata`.

## Einschränkungen

Vier Familien, nur Objekterkennung: `yolox`, `yolo9`, `rtdetr` und `rfdetr`. Alles
andere wird im Preflight abgelehnt, denn erst der familienbewusste
Preprocessing-Wrapper macht den festen Vertrag für die Bildeingabe korrekt, und
eine Familie außerhalb davon würde mit der falschen Normalisierung konvertiert.
Die Fehlermeldung nennt ONNX und TorchScript als Alternativen.

Die Eingabeform ist durch `ct.ImageType` fest verdrahtet, `dynamic=True` ändert
also nichts und die Metadaten notieren `dynamic=False`. Exportiere für eine zweite
Auflösung ein zweites Bundle.

`half=True` konvertiert mit FP16-Präzision. Einen INT8-Pfad gibt es in diesem
Exporter nicht.

Das vollständige Raster aus Familien und Aufgaben zeigt
[die Export-Matrix](/docs/reference/export-matrix). Apples neueres
On-Device-Format beschreibt [Core AI](/docs/export/coreai). Für eine einzelne
Kombination:

<code-tabs name="support" />
