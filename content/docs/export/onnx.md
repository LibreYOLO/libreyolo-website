---
title: ONNX
seo_title: "Export to ONNX from LibreYOLO"
description: "Export a LibreYOLO model to ONNX: the opset LibreYOLO picks per family, dynamic axes, embedded NMS, INT8, and how the graph loads back."
lead: "ONNX is a portable graph format. LibreYOLO traces the model with torch.onnx.export, optionally simplifies the graph, and writes the family, task, class names and input size into the file's own metadata so any LibreYOLO backend can rebuild the postprocessing."
keywords:
  - yolo onnx export
  - onnxruntime
  - torch.onnx.export
  - onnx opset
  - dynamic axes
  - embedded nms onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="onnx")'
    mono: true
  - label: Writes
    value: "One .onnx file, metadata embedded in the graph"
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t.onnx")'
    mono: true
  - label: Shapes
    value: "Dynamic batch by default in Python; per-task exceptions below"
  - label: Precision
    value: "FP32, FP16 (half=True), INT8 (int8=True, YOLO9 detection)"
verification: "Read from libreyolo/export/onnx.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/onnx.py and libreyolo/cli/commands/export.py on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Arguments
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int, or (height, width)
            batch=1,
            dynamic=True,     # Python default; the CLI defaults to False
            simplify=True,    # run onnxsim over the graph
            opset=None,       # None picks 13, or 17 for DETR-style families
            half=False,       # FP16 weights and activations
            int8=False,       # QDQ INT8, YOLO9 detection only
            data=None,        # calibration data.yaml, INT8 only
            device=None,      # trace device; None uses the model's device
            output_path=None, # None writes weights/<stem>.onnx
        )
  nms:
    - label: Embed NMS in the graph
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLO9 detection only, batch 1. dynamic is forced to False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 with calibration data
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # a few hundred representative images
            fraction=1.0,
        )
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Bare ONNX Runtime
      language: python
      code: |
        import numpy as np
        import onnx
        import onnxruntime as ort

        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )

        # Preprocessing and postprocessing are yours on this path.
        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: batch})
        print([out.shape for out in outputs])

        # The graph carries the family, task, class names and input size.
        meta = {p.key: p.value for p in onnx.load("weights/LibreYOLO9t.onnx").metadata_props}
        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

<code-tabs name="install" />

The extra pulls `onnx`, `onnxsim` and `onnxruntime`. `onnx` alone is enough to
write the file; `onnxsim` runs the simplification pass and `onnxruntime` runs the
artifact and performs INT8 calibration.

## Export

<code-tabs name="export" />

Without `output_path`, the file lands in `weights/` under the checkpoint's stem,
with `_fp16` or `_int8` appended when that precision was requested.

`dynamic` defaults to `True` in Python and `False` on the CLI. When it is on, the
batch axis becomes symbolic and a few tasks widen further: semantic segmentation
also opens the mask height and width, Real-ESRGAN restoration opens the spatial
axes, and the two-stage detectors keep source height and width dynamic because
their resize happens inside the graph.

`opset` is chosen per family when omitted. DETR-style families (`detr`,
`deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`,
`rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) plus `deit`, `midas` and `moge2` get
opset 17, which is where `aten::scaled_dot_product` lowers. Everything else gets
13. Matting is raised to 19 regardless, because BiRefNet's decoder needs the
`DeformConv` operator, which ONNX defines from opset 19.

`simplify=True` runs `onnxsim` and keeps the original graph if the pass fails, so
a simplification error is a warning rather than an export failure. On macOS arm64
with `onnx` 1.22 or newer and `onnxsim` 0.6.5 or older the pass is skipped
entirely, because that pairing can abort the Python process.

### Embedded NMS

<code-tabs name="nms" />

`nms=True` is YOLO9 detection only and requires batch 1; requesting it with
`dynamic=True` logs a warning and turns dynamic off. The graph then has two
outputs: `output`, shaped `(batch, max_det, 6)`, and `raw`, the undecoded
detector tensor that LibreYOLO's own backend uses so postprocessing stays
identical to the PyTorch path.

### INT8

<code-tabs name="int8" />

`int8=True` runs ONNX Runtime static quantization and writes a QDQ graph with
float32 inputs and outputs. Only `Conv` and `Gemm` nodes are quantized. Leaving
the detection-head decode in float32 is deliberate: that concatenation mixes
pixel-scale box coordinates with class scores in the range 0 to 1, and a single
per-tensor activation scale dominated by the box magnitude would drive every
score to zero.

This flag currently applies to YOLO9 detection only, and anything else raises
`NotImplementedError` in preflight. Omitting `data` falls back to `coco8.yaml`
with a warning; eight images is not a representative calibration set. A model
that was already quantized in PyTorch takes a different route, described on
[Quantization](/docs/export/quantization).

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` dispatches on the `.onnx` suffix and returns the same `Results`
object as a `.pt` checkpoint, because the class names, task, input size and pose
schema were written into the graph's `metadata_props` at export time. With
`device="auto"` the session takes `CUDAExecutionProvider` when ONNX Runtime
reports it and falls back to CPU otherwise.

The second snippet is for readers with no LibreYOLO installed. Preprocessing,
decoding, NMS and coordinate rescaling all become yours on that path; the
metadata block is still there to read.

## Constraints

Output tensor names are fixed per task, and they are what the metadata-free
consumer has to match:

| Task | Output names |
|---|---|
| Detection, grid and anchor heads | `output` |
| Detection, DETR-style | `pred_logits`, `pred_boxes` |
| Detection, RF-DETR | `dets`, `labels` |
| Classification | `output` |
| Semantic segmentation | `semantic_logits` |
| Depth | `depth` |
| Surface normal | `normal` |
| Edge | `edges` |
| Restoration | `restored` |
| Matting | `matte` |
| Gaze | `yaw_logits`, `pitch_logits` |

RF-DETR is also the one family whose input tensor is named `input` rather than
`images`.

Several tasks carry a fixed-resolution runtime contract in this version. Depth,
surface normal and edge reject `batch != 1` and force `dynamic=False`. Matting
forces the native 1024 square, because BiRefNet's Swin relative-position tables
are tied to their resolution. Restoration forces a fixed canvas for every family
except Real-ESRGAN, whose generator is fully convolutional.

Rectangular `imgsz` works for the YOLO9 families, HRNet, NAFNet and Real-ESRGAN.
Families with a fixed square contract (`clip`, `deformable_detr`, `detr`,
`dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `moge2`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`) reject it outright.

Two combinations are refused before tracing: YOLO9 segmentation, because YOLO9 is
detection only in LibreYOLO, and RTMDet-Ins segmentation, whose dynamic-kernel
mask decode has no exported-runtime contract.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination, ask the
library directly:

<code-tabs name="support" />
