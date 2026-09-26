---
title: Core ML
seo_title: "Export to Core ML from LibreYOLO"
description: "Export a LibreYOLO detector to a Core ML .mlpackage: the ImageType input contract, FP16, compute units, embedded NMS, and the four supported families."
lead: "Core ML is Apple's on-device model format. LibreYOLO traces the detector behind a per-family preprocessing wrapper so the converted graph always takes a canonical RGB image input, then writes an ML Program .mlpackage with the model metadata attached."
keywords:
  - yolo coreml export
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms pipeline
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="coreml")'
    mono: true
  - label: Writes
    value: "One .mlpackage bundle (a directory) in ML Program format"
  - label: Extra
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t.mlpackage") on macOS'
    mono: true
  - label: Shapes
    value: "Fixed. The input is a hard-shaped ct.ImageType."
  - label: Precision
    value: "FP32, FP16 (half=True). No INT8."
  - label: Families
    value: "Detection only, for yolox, yolo9, rtdetr and rfdetr"
verification: "Read from libreyolo/export/coreml.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/coreml.py and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes the bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Arguments
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True converts with FLOAT16 compute precision
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None writes weights/<stem>.mlpackage
        )

        # dynamic is accepted but the input is a fixed-shape ct.ImageType,
        # and the embedded metadata records dynamic=False either way.
  nms:
    - label: Embed Apple's NMS layer
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX and YOLO9 detection only, batch 1.
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
    - label: Through LibreYOLO, on macOS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # or cpu_and_ne to pin the Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Bare coremltools
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # The input is an image named "image" at the fixed export size.
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # Letterboxing and postprocessing are yours on this path.
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

<code-tabs name="install" />

Prediction needs macOS. `LibreYOLO()` refuses a `.mlpackage` on any other platform
with a message naming the current one, and the support matrix records these
combinations as available on the grounds that runtime parity needs a macOS runner.

## Export

<code-tabs name="export" />

The bundle is written to `weights/` under the checkpoint's stem, with `_fp16`
appended when `half=True`. A `.mlpackage` is a directory, so copy the whole tree.

Every family is traced behind a preprocessing wrapper, so the converted graph
takes one canonical input: RGB, `scale=1/255`, no bias, declared as
`ct.ImageType`. The wrapper absorbs the family's own convention, which is BGR in
the range 0 to 255 for YOLOX, ImageNet mean and standard deviation for RF-DETR,
and identity for YOLO9 and RT-DETR. That is why a Core ML consumer feeds an
ordinary image rather than a family-specific tensor.

Conversion targets ML Program with a minimum deployment target of iOS 15.
`compute_units` is stored on the converted model and can be overridden again when
the artifact is loaded.

Model metadata goes into `user_defined_metadata` as strings, which is where the
backend reads the family, task, class names, input size and pose schema.

### Embedded NMS

<code-tabs name="nms" />

`nms=True` wraps the model in a Core ML pipeline that ends in Apple's
`NonMaximumSuppression` layer. The result has two outputs: `confidence`, shaped
`N` by the class count, and `coordinates`, shaped `N` by 4 as normalized `xywh`.

It applies to YOLOX and YOLO9 detection only, and it requires batch 1. The
DETR-style families are refused by name, because set prediction takes a top-k over
queries and classes with no IoU step and cannot use that layer. `max_det` is not
exposed here either; when the detection cap matters, use
[ONNX embedded NMS](/docs/export/onnx) instead.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` recognizes a directory with the `.mlpackage` suffix and returns the
same `Results` object as the checkpoint. `compute_units` is the one argument the
factory passes through for this format, and it accepts `all`, `cpu_and_gpu`,
`cpu_and_ne` and `cpu_only`. The `device` argument is ignored, because Core ML
routes through compute units instead.

The second snippet is the bare-runtime path. Letterboxing, decoding, NMS and
coordinate rescaling become yours there, and the class names live in
`user_defined_metadata`.

## Constraints

Four families, detection only: `yolox`, `yolo9`, `rtdetr` and `rfdetr`. Anything
else is refused in preflight, because the family-aware preprocessing wrapper is
what makes the fixed image input contract correct, and a family outside it would
convert with the wrong normalization. The error names ONNX and TorchScript as the
alternatives.

The input shape is hard-fixed by `ct.ImageType`, so `dynamic=True` changes nothing
and the metadata records `dynamic=False`. Export a second bundle for a second
resolution.

`half=True` converts with FP16 compute precision. There is no INT8 path from this
exporter.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For Apple's newer on-device
format, see [Core AI](/docs/export/coreai). For one combination:

<code-tabs name="support" />
