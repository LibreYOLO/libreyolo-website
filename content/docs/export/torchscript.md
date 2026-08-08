---
title: TorchScript
seo_title: "Export to TorchScript from LibreYOLO"
description: "Export a LibreYOLO model to TorchScript: a traced .torchscript archive with LibreYOLO metadata inside, loadable from Python or libtorch."
lead: "TorchScript is PyTorch's own serialized-graph format. LibreYOLO traces the model with torch.jit.trace and saves the result together with a libreyolo_metadata.json extra file, so the archive carries the family, task, class names and input size."
keywords:
  - yolo torchscript export
  - torch.jit.trace
  - torch.jit.load
  - libtorch deployment
  - torchscript metadata
  - extra_files
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="torchscript")'
    mono: true
  - label: Writes
    value: "One .torchscript archive with a libreyolo_metadata.json extra file"
  - label: Extra
    value: "None. TorchScript ships with PyTorch."
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t.torchscript")'
    mono: true
  - label: Shapes
    value: "Fixed. The graph is traced at one input shape."
  - label: Precision
    value: "FP32, FP16 (half=True). No INT8."
verification: "Read from libreyolo/export/torchscript.py, libreyolo/export/exporter.py, libreyolo/export/support.py and libreyolo/backends/torchscript.py on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Arguments
      language: python
      code: |
        model.export(
            format="torchscript",
            imgsz=640,        # int, or (height, width)
            batch=1,
            half=False,       # FP16 weights and activations
            device=None,      # None traces on CPU for this format
            output_path=None, # None writes weights/<stem>.torchscript
        )

        # dynamic is accepted but the archive is always a fixed-shape trace,
        # and the embedded metadata records dynamic=False either way.
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Bare PyTorch
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

        # Preprocessing and postprocessing are yours on this path.
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

<code-tabs name="install" />

TorchScript needs nothing beyond the base install, because `torch.jit` ships with
PyTorch. It is the one export target with no optional dependency and no external
converter, which makes it a useful first check when a longer toolchain fails.

## Export

<code-tabs name="export" />

Tracing runs on CPU unless a device is named, and the archive is written to
`weights/` under the checkpoint's stem when `output_path` is omitted.

The retrace check that `torch.jit.trace` normally performs is turned off. Several
export wrappers cache shape-dependent anchors during their first forward pass, so
a second trace observes a different Python path even though the recorded
fixed-shape graph is correct. Parity tests validate the saved module directly
instead.

Metadata does not live in a sidecar. `torch.jit.save` stores
`libreyolo_metadata.json` inside the archive, and `torch.jit.load` hands it back
through `_extra_files`.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` routes on the `.torchscript` suffix and returns the same `Results`
object as the checkpoint it came from. With `device="auto"` the module is mapped
to CUDA when available, then MPS, then CPU.

The second snippet is the path for a reader with no LibreYOLO installed, and for
C++ deployment through libtorch, where the same archive loads with
`torch::jit::load`. Preprocessing, decoding, NMS and coordinate rescaling become
yours there. The metadata extra file is still readable, and it is the only place
the class names exist.

## Constraints

The graph is a trace at one input shape. `dynamic=True` is accepted for interface
symmetry but changes nothing, and the embedded metadata reports `dynamic=False`
so a backend never assumes an axis it cannot use. Export a second archive for a
second resolution.

`half=True` casts the model and the trace input to FP16. There is no INT8 path:
`int8=True` raises `NotImplementedError` during validation.

Rectangular `imgsz` works for the YOLO9 families, HRNet, NAFNet and Real-ESRGAN,
and is rejected for families with a fixed square contract.

Five combinations are refused before tracing. YOLO9 segmentation, because YOLO9
is detection only in LibreYOLO. RTMDet-Ins segmentation, whose dynamic-kernel
mask decode has no exported-runtime contract. SSD, Faster R-CNN and RetinaNet
detection, whose variable-length or dynamic-anchor graphs have parity evidence
only through the ONNX Runtime contract.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination:

<code-tabs name="support" />
