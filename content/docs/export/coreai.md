---
title: Core AI
seo_title: "Export to Apple Core AI from LibreYOLO"
description: "Export a LibreYOLO model to an Apple Core AI .aimodel asset: macOS only, fixed canvas, FP32, and the named-output ordering contract consumers must respect."
lead: "Core AI is Apple's on-device inference stack. LibreYOLO captures the model with torch.export, lowers it through the Core AI converter, and writes a .aimodel asset carrying the model metadata and the exported output names."
keywords:
  - libreyolo core ai export
  - aimodel
  - coreai-torch
  - torch.export apple
  - apple on-device inference
  - coreai_output_names
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="coreai")'
    mono: true
  - label: Writes
    value: "One .aimodel asset with metadata attached"
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Loads back
    value: "Not through LibreYOLO. Consumers use the Core AI runtime directly."
  - label: Shapes
    value: "Fixed canvas. dynamic=True raises NotImplementedError."
  - label: Precision
    value: "FP32 only. half=True and int8=True are rejected."
  - label: Requires
    value: "macOS. The toolchain neither converts nor runs elsewhere, and coreai-torch pins torch to 2.11.x."
verification: "Read from libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py, libreyolo/export/exporter.py, libreyolo/export/support.py and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install, on macOS
      language: bash
      code: |
        # Kept out of every aggregate extra on purpose: coreai-torch pins torch
        # to 2.11.x and would drag the whole environment onto that version.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int, or (height, width); this is the run canvas
            batch=1,
            output_path=None, # None writes weights/<stem>.aimodel
        )

        # dynamic=True raises NotImplementedError.
        # half=True and int8=True are rejected during validation.
  outputs:
    - label: Read the output ordering before wiring a consumer
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # The asset metadata records the exported output names, in graph order,
        # under "coreai_output_names". Map Core AI's returned dictionary by name
        # using that list; never pair it positionally with the eager tuple.
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

This format is macOS only. The `coreai-torch` requirement carries a
`sys_platform == 'darwin'` marker, and the toolchain neither converts nor runs
anywhere else.

<code-tabs name="install" />

The extra sits outside every aggregate extra, including `libreyolo[all]`, because
`coreai-torch` pins torch to the 2.11 series. Install it into an environment you
are willing to constrain to that pair.

## Export

<code-tabs name="export" />

Capture is `torch.export`, a real graph capture with guards, rather than a single
recorded trace. That is stricter than the Core ML path: host scalar reads and
data-dependent control flow are rejected instead of being silently baked in, which
is why a few families are blocked here with a capture failure recorded against
them.

Three preparation steps run inside a scope that restores the caller's live model
whether the export succeeds or fails. Darknet-derived families get their inference
batch normalization folded exactly into the preceding convolutions, because Core AI
0.4.1 does not preserve Darknet's epsilon-after-square-root formula. Grid and
anchor families get their anchors frozen for the fixed canvas. RF-DETR gets its
position embedding rebaked for the requested canvas by re-running the model's own
baking path, because the converter has no lowering for
`aten._upsample_bicubic2d_aa`.

Lowering folds PyTorch's reference decomposition for `aten.grid_sampler_2d` into
the decomposition table, since the Core AI converter has no lowering for the
deformable-attention sampler the DETR families use.

Assets declare a minimum OS of v27, which is the only value the toolchain offers.
That gates deployment, not conversion: conversion and Python-side execution work
on earlier macOS through the runtime inside the wheel, but numerics differ between
OS versions, so recorded parity is measured on macOS 27.

## Run the artifact

There is no Core AI entry in `libreyolo/backends`, so `LibreYOLO()` does not load
a `.aimodel`. Consumers use the Core AI runtime directly, and preprocessing,
decoding, NMS and coordinate rescaling are theirs. A validated row in the support
matrix is a claim that the exported graph computes the same numbers as the
reference, not that `predict` will run it.

The one thing a consumer cannot re-derive is the output ordering:

<code-tabs name="outputs" />

Core AI returns a named dictionary whose key order matches neither the eager
forward's tuple order nor anything guessable. The exported names are written into
the asset metadata as `coreai_output_names` for exactly this reason. Map by name.

## Constraints

Fixed canvas, FP32, batch as exported. `dynamic=True` raises
`NotImplementedError`, and `half=True` and `int8=True` are rejected during
validation.

Coverage is wide on the conversion side. Validated combinations include the YOLO9
families, YOLOX, YOLO7, the four Darknet-era detectors, YOLO-NAS, PicoDet, RTMDet,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2, EC and RF-DETR detection; the
four CNN classification families plus frozen-class CLIP and SigLIP2; Depth
Anything V2 and ZipDepth; NAFNet and Real-ESRGAN restoration; PIDNet and
LingBotVision semantic segmentation; and FOMO point detection. Each carries its
own recorded context, which `libreyolo formats` prints.

Blocked, with the reason recorded per combination:

| Combination | Why |
|---|---|
| EoMT semantic segmentation | Strict capture fails with `GuardOnDataDependentSymNode`: something in the mask path reads a value off a tensor and branches on it |
| SegFormer semantic segmentation | The capture path has not been assessed, and its published weights are non-commercial regardless of format |
| L2CS gaze | The model itself supports ONNX, TorchScript, ExecuTorch, TensorRT and OpenVINO only, which is a model-side decision |
| Depth Anything 3 depth | The family rejects export for every format |

RF-DETR carries one caveat worth reading before comparing artifacts. Its parity is
recorded against the graph the Core AI exporter itself prepares, not against ONNX,
and at a 640 canvas the RF-DETR ONNX artifact disagrees with that prepared graph.
The Core AI rebake preserves the antialiased resize the eager model performs,
while the ONNX path disables antialiasing. ONNX is therefore not a valid reference
for that family at a non-native canvas.

For Apple's earlier format, see [Core ML](/docs/export/coreml). For the full
family and task grid, see [the export matrix](/docs/reference/export-matrix). For
one combination:

<code-tabs name="support" />
