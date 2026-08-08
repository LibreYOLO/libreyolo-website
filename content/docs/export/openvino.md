---
title: OpenVINO
seo_title: "Export to OpenVINO IR from LibreYOLO"
description: "Convert a LibreYOLO model to OpenVINO IR: the model.xml and model.bin pair, FP16 weight compression, NNCF INT8, and CPU, GPU or NPU inference."
lead: "OpenVINO IR is Intel's runtime format, a model.xml graph beside a model.bin weight blob. LibreYOLO exports an ONNX intermediate, converts it with ov.convert_model, and writes a metadata.yaml into the same directory."
keywords:
  - yolo openvino export
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - nncf int8 quantization
  - openvino npu
  - compress_to_fp16
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="openvino")'
    mono: true
  - label: Writes
    value: "A directory with model.xml, model.bin and metadata.yaml"
  - label: Extra
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t_openvino")'
    mono: true
  - label: Shapes
    value: "Follows the ONNX intermediate: dynamic batch when dynamic=True"
  - label: Precision
    value: "FP32, FP16 weight compression (half=True), INT8 via NNCF (int8=True with data=)"
verification: "Read from libreyolo/export/openvino.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/openvino.py and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        # The IR is converted from an ONNX intermediate, so both extras are needed.
        pip install "libreyolo[onnx,openvino]"
    - label: INT8 additionally needs NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes the directory weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Arguments
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True keeps a dynamic batch axis through the IR
            half=False,       # True stores FP16 weights
            int8=False,       # True runs NNCF post-training quantization
            data=None,        # required when int8=True
            output_path=None, # None writes weights/<stem>_openvino
        )
  int8:
    - label: INT8 with calibration data
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # required: there is no default for this format
            fraction=1.0,
        )
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Select the device
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" and "cpu" map to CPU, "gpu" and "cuda" map to GPU,
        # anything else is passed through uppercased, for example "npu" -> NPU.
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: Bare OpenVINO
      language: python
      code: |
        import numpy as np
        import openvino as ov
        import yaml

        core = ov.Core()
        print(core.available_devices)

        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml", "CPU")
        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))
        print([tensor.shape for tensor in outputs.values()])

        # Class names, task and input size live in metadata.yaml beside the IR.
        meta = yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Preprocessing and postprocessing are yours on this path.
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

<code-tabs name="install" />

The conversion goes through an ONNX intermediate, so the `onnx` extra is part of
the requirement rather than an optional companion. NNCF is a separate install and
is only needed for `int8=True`.

## Export

<code-tabs name="export" />

The artifact is a directory, not a file. `weights/LibreYOLO9t_openvino` holds
`model.xml`, `model.bin` and `metadata.yaml`, and `_fp16` is inserted before the
suffix when `half=True`. Move or copy the whole directory; the three files are one
artifact.

`half=True` sets `compress_to_fp16` on save. That is weight compression in the IR,
not a change to the inference precision the device chooses at run time.

### INT8

<code-tabs name="int8" />

`int8=True` runs NNCF post-training quantization over a LibreYOLO calibration
loader with the mixed preset, and `data` is mandatory: this format has no
eight-image fallback. Missing NNCF raises an `ImportError` naming the install
command.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` recognizes any directory containing `model.xml` and returns the same
`Results` object as the checkpoint, reading class names, task, input size and
pose schema from `metadata.yaml`.

The device string is mapped rather than passed straight through. `auto` and `cpu`
both compile for CPU, `gpu` and `cuda` both compile for GPU, and any other value
is uppercased and handed to OpenVINO, which is how an NPU target is reached.

The third snippet is for readers with no LibreYOLO installed. Preprocessing,
decoding, NMS and coordinate rescaling become yours there, and the class names
only exist in `metadata.yaml`.

## Constraints

An IR without its `metadata.yaml` still loads, but the backend then falls back to
80 classes and the detection task, which is wrong for anything else. Keep the
directory intact.

Blocked before tracing: YOLO9 segmentation, RTMDet-Ins segmentation, SSD,
Faster R-CNN and RetinaNet detection, and BiRefNet or FeyNobg matting, where
OpenVINO 2026.2 cannot lower the shared matte decoder's standard ONNX
`DeformConv-19` operation.

Where a combination is neither validated nor blocked, the converter path is
available and the project has not recorded OpenVINO runtime parity for it. Several
combinations are validated with an explicit context attached, for example
DeepLabV3 semantic segmentation at a fixed 520 by 520 input on OpenVINO 2026.2
with the CPU default inference precision, and L2CS gaze at a fixed 448 by 448
face crop. `libreyolo formats` prints that context per combination.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination:

<code-tabs name="support" />
