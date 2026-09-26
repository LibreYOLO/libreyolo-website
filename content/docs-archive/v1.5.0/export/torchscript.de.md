---
title: TorchScript
seo_title: Export nach TorchScript aus LibreYOLO
description: >-
  Ein LibreYOLO-Modell nach TorchScript exportieren: ein per Tracing erzeugtes
  .torchscript-Archiv mit LibreYOLO-Metadaten darin, ladbar aus Python oder aus
  libtorch.
lead: >-
  TorchScript ist PyTorchs eigenes Format für serialisierte Graphen. LibreYOLO
  zeichnet das Modell mit torch.jit.trace auf und speichert das Ergebnis
  zusammen mit einer libreyolo_metadata.json als Extra-Datei, sodass das Archiv
  Familie, Aufgabe, Klassennamen und Eingabegröße mitführt.
keywords:
  - yolo nach torchscript exportieren
  - torch.jit.trace
  - torch.jit.load
  - libtorch deployment
  - torchscript metadaten
  - extra_files
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="torchscript")
    mono: true
  - label: Schreibt
    value: Ein .torchscript-Archiv mit einer libreyolo_metadata.json als Extra-Datei
  - label: Extra
    value: Keins. TorchScript gehört zu PyTorch.
  - label: Lädt zurück
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Shapes
    value: Fest. Der Graph wird bei genau einer Eingabe-Shape aufgezeichnet.
  - label: Präzision
    value: 'FP32, FP16 (half=True). Kein INT8.'
verification: >-
  Gelesen aus libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py und libreyolo/backends/torchscript.py auf dem
  dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Argumente
      language: python
      code: >
        model.export(
            format="torchscript",
            imgsz=640,        # int, oder (Höhe, Breite)
            batch=1,
            half=False,       # FP16-Gewichte und -Aktivierungen
            device=None,      # None traced bei diesem Format auf der CPU
            output_path=None, # None schreibt weights/<stem>.torchscript
        )


        # dynamic wird akzeptiert, aber das Archiv ist immer ein Trace mit
        fester

        # Shape, und die eingebetteten Metadaten halten so oder so dynamic=False
        fest.
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Reines PyTorch
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # Pre- und Postprocessing liegen auf diesem Weg bei dir.
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Eine Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Installation

<code-tabs name="install" />

TorchScript braucht nichts über die Basisinstallation hinaus, weil `torch.jit` zu
PyTorch gehört. Es ist das eine Exportziel ohne optionale Abhängigkeit und ohne
externen Konverter, was es zu einer nützlichen ersten Probe macht, wenn eine
längere Toolchain scheitert.

## Export

<code-tabs name="export" />

Das Tracing läuft auf der CPU, solange kein Gerät angegeben ist, und das Archiv
wird unter dem Stamm des Checkpoints nach `weights/` geschrieben, wenn
`output_path` weggelassen wird.

Die Retrace-Prüfung, die `torch.jit.trace` normalerweise durchführt, ist
abgeschaltet. Mehrere Export-Wrapper legen beim ersten Forward-Pass
shape-abhängige Anchors im Cache ab, sodass ein zweiter Trace einen anderen
Python-Pfad beobachtet, obwohl der aufgezeichnete Graph mit fester Shape korrekt
ist. Die Paritätstests prüfen stattdessen direkt das gespeicherte Modul.

Die Metadaten liegen nicht in einer Sidecar-Datei. `torch.jit.save` legt
`libreyolo_metadata.json` im Archiv selbst ab, und `torch.jit.load` gibt sie über
`_extra_files` zurück.

## Ausführen des Artefakts

<code-tabs name="run" />

`LibreYOLO()` entscheidet anhand der Endung `.torchscript` und liefert dasselbe
`Results`-Objekt wie der Checkpoint, aus dem es stammt. Mit `device="auto"` wird
das Modul auf CUDA gemappt, sofern verfügbar, danach auf MPS, danach auf die CPU.

Das zweite Snippet ist der Weg für Leser ohne installiertes LibreYOLO und für das
Deployment in C++ über libtorch, wo dasselbe Archiv mit `torch::jit::load` lädt.
Preprocessing, Decoding, NMS und die Skalierung der Koordinaten liegen dort bei
dir. Die Metadaten-Extra-Datei bleibt lesbar, und sie ist der einzige Ort, an dem
die Klassennamen stehen.

## Einschränkungen

Der Graph ist ein Trace bei genau einer Eingabe-Shape. `dynamic=True` wird der
Symmetrie der Schnittstelle halber akzeptiert, ändert aber nichts, und die
eingebetteten Metadaten melden `dynamic=False`, damit ein Backend nie eine Achse
annimmt, die es nicht nutzen kann. Für eine zweite Auflösung exportierst du ein
zweites Archiv.

`half=True` castet das Modell und die Trace-Eingabe nach FP16. Einen INT8-Weg gibt
es nicht: `int8=True` löst bei der Validierung einen `NotImplementedError` aus.

Ein rechteckiges `imgsz` funktioniert für die YOLO9-Familien, HRNet, NAFNet und
Real-ESRGAN und wird für Familien mit festem quadratischem Kontrakt abgelehnt.

Fünf Kombinationen werden vor dem Tracing abgelehnt. Die YOLO9-Segmentierung, weil
YOLO9 in LibreYOLO ausschließlich Erkennung ist. Die RTMDet-Ins-Segmentierung,
deren Masken-Decoding mit dynamischen Kernels keinen Kontrakt für exportierte
Runtimes hat. Die Erkennung mit SSD, Faster R-CNN und RetinaNet, deren Graphen mit
variabler Länge oder dynamischen Anchors nur über den Kontrakt der ONNX Runtime
Paritätsbelege haben.

Das vollständige Raster aus Familien und Aufgaben steht in
[der Export-Matrix](/docs/reference/export-matrix). Für eine einzelne Kombination:

<code-tabs name="support" />
