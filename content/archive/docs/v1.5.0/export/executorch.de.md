---
title: ExecuTorch
seo_title: Export nach ExecuTorch aus LibreYOLO
description: >-
  Ein LibreYOLO-Modell in ein ExecuTorch-.pte-Programm mit XNNPACK-Delegation
  exportieren: feste Shape, Batch 1, FP32 und die Metadaten-Sidecar-Datei, die
  es braucht.
lead: >-
  ExecuTorch führt PyTorch-Programme auf Edge-Zielen aus. LibreYOLO erfasst das
  Modell mit torch.export im Strict-Modus, überführt es per Lowering auf XNNPACK
  und legt das .pte-Programm zusammen mit einer JSON-Metadaten-Sidecar-Datei als
  eine Einheit ab.
keywords:
  - yolo executorch export
  - .pte programm
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - pytorch inferenz auf edge-geräten
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="executorch")
    mono: true
  - label: Schreibt
    value: Ein .pte-Programm plus eine .pte.json-Metadaten-Sidecar-Datei
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Lädt zurück
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: Shapes
    value: Fest. dynamic=True und batch != 1 werden abgelehnt.
  - label: Präzision
    value: Nur FP32. half=True und int8=True werden abgelehnt.
  - label: Delegate
    value: 'XNNPACK, CPU. delegate=''xnnpack'' ist der einzige akzeptierte Wert.'
verification: >-
  Gelesen aus libreyolo/export/executorch.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/executorch.py und
  pyproject.toml auf dem dev-Branch.
snippets:
  install:
    - label: Installation
      language: bash
      code: |
        # Bewusst nicht in libreyolo[all]: ExecuTorch schränkt ein, mit welcher
        # Torch-Version es kombiniert werden kann.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Schreibt weights/LibreYOLO9t.pte und weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Argumente
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, oder (Höhe, Breite)
            batch=1,               # jeder andere Wert löst ValueError aus
            dynamic=False,         # True löst ValueError aus
            delegate="xnnpack",    # der einzige akzeptierte Wert
            device="cpu",          # jedes andere Gerät löst ValueError aus
            output_path=None,      # None schreibt weights/<stem>.pte
        )
  run:
    - label: Über LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Reine ExecuTorch-Runtime
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # Pre- und Postprocessing liegen auf diesem Weg bei dir.

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Eine Familie und Aufgabe vor dem Export prüfen
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## Installation

<code-tabs name="install" />

Dieses Extra liegt bewusst außerhalb von `libreyolo[all]`, weil ExecuTorch
festlegt, mit welcher Torch-Version es funktioniert, und eine Installation die
gesamte Umgebung auf diese Paarung ziehen würde. Installiere es in eine Umgebung,
die du bereit bist einzuschränken.

Unter Windows ruft der Lowering-Schritt die ausführbare Datei `flatc` auf, die
ExecuTorch mitliefert. Liegt sie nicht im `PATH`, löst der Export einen
`RuntimeError` mit genau diesem Hinweis aus, und der Ausweg ist, aus einer
Developer PowerShell von Visual Studio 2022 heraus zu starten.

## Export

<code-tabs name="export" />

Erfasst wird mit `torch.export.export(..., strict=True)`, also eine echte
Graph-Erfassung mit Guards statt eines aufgezeichneten Trace. Lesezugriffe auf
Host-Skalare und datenabhängiger Kontrollfluss werden abgelehnt, statt still
eingebacken zu werden, deshalb scheitern hier mehrere Familien, deren Tracing
anderswo gelingt; die Gründe stehen pro Kombination in der Support-Matrix.

Das Lowering führt `to_edge_transform_and_lower` mit dem XNNPACK-Partitioner aus.
Enthält das Ergebnis null Delegate-Partitionen, bricht der Export mit einem Fehler
ab, statt ein Programm aus reinen Portable-Kernels als XNNPACK auszuweisen.

Programm und Sidecar-Datei werden gemeinsam festgeschrieben. Beide werden vorab
abgelegt, beide werden eingewechselt, und ein Fehler setzt auf den vorherigen
Stand zurück, ein halbes Paar landet also nie auf der Festplatte.

## Ausführen des Artefakts

<code-tabs name="run" />

`LibreYOLO()` entscheidet anhand der Endung `.pte` und liefert dasselbe
`Results`-Objekt wie der Checkpoint. Die Sidecar-Datei ist beim Laden Pflicht:
ohne `<program>.pte.json` löst das Backend einen `FileNotFoundError` aus, weil das
Programm selbst keine Klassennamen, keine Aufgabe und keine Eingabegröße
mitbringt. Das Backend prüft vor dem Laden außerdem, ob die installierte Runtime
`XnnpackBackend` bereitstellt, und liest das Programm aus Bytes, statt die Datei zu
mappen, was verhindert, dass unter Windows für die Lebensdauer des Backends eine
Dateisperre gehalten wird.

Das zweite Snippet ist der Weg über die reine Runtime. Preprocessing, Decoding,
NMS und die Skalierung der Koordinaten liegen dort bei dir.

## Einschränkungen

Batch 1, feste Shape, FP32, CPU. `batch != 1` und `dynamic=True` lösen beide einen
`ValueError` aus, bevor der Export irgendetwas verändert, `half=True` und
`int8=True` werden bei der Validierung abgelehnt, und ein anderes Gerät als die CPU
wird verweigert.

`delegate` akzeptiert in dieser Version `"xnnpack"` und sonst nichts.

Exporte für die Klassifikation tragen zwei zusätzliche Metadaten-Schlüssel,
`crop_pct` und `interpolation`, damit die Runtime das Resize- und
Center-Crop-Verhalten der Familie reproduzieren kann.

Die blockierten Einträge benennen den konkreten Fehler statt einer Kategorie.
D-FINE-Erkennung und -Segmentierung stoßen unter der strikten Erfassung auf einen
nicht unterstützten `ContextVar`-Lesezugriff in der Deformable Attention, und der
erzwungene manuelle Grid-Sample-Weg serialisiert zwar, scheitert dann aber zur
Laufzeit an einer ungültigen Dimensionsreihenfolge des delegierten Tensors. DEIM
und DEIMv2 durchlaufen Erfassung, Lowering und Serialisierung und scheitern dann
während der Ausführung. Die semantische Segmentierung von EoMT scheitert an einem
datenabhängigen symbolischen Ausdruck im Masken-Pfad. BiRefNet-Matting wird bei
1024 mal 1024 erfasst, hat aber keine Out-Variante für
`torchvision::deform_conv2d`. Die SwinIR-Restaurierung lädt zwar wieder, scheitert
dann aber in `aten::alias_copy.out` an nicht zusammenpassenden
Dimensionsreihenfolgen.

Das vollständige Raster aus Familien und Aufgaben steht in
[der Export-Matrix](/docs/reference/export-matrix). Für eine einzelne Kombination:

<code-tabs name="support" />
