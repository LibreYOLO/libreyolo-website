---
title: TFLite
seo_title: "Export to TFLite (LiteRT) from LibreYOLO"
description: "Export a LibreYOLO model to a .tflite FlatBuffer through onnx2tf: static shapes, FP32 only, NHWC inputs, and the families that convert cleanly."
lead: "TFLite is the FlatBuffer format LiteRT executes on mobile and embedded targets. LibreYOLO exports a static ONNX graph, converts it with onnx2tf in flatbuffer-direct mode, and writes the model metadata beside the artifact as a JSON sidecar."
keywords:
  - yolo tflite export
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - nhwc input tflite
  - edge inference
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="tflite")'
    mono: true
  - label: Writes
    value: "One .tflite file plus a .tflite.json metadata sidecar"
  - label: Extra
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t.tflite")'
    mono: true
  - label: Shapes
    value: "Static only. dynamic=True is rejected."
  - label: Precision
    value: "FP32 only. half=True and int8=True are rejected."
  - label: Requires
    value: "Python 3.12 or newer, because onnx2tf 2.4.x publishes no older wheels"
verification: "Read from libreyolo/export/tflite.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/tflite.py and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        # LiteRT is Google's current name for TensorFlow Lite. Both extras
        # install the same toolchain and produce the same .tflite output.
        pip install "libreyolo[tflite]"
    - label: Confirm the Python version first
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes weights/LibreYOLO9t.tflite and weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" is accepted as an alias and resolves to the same exporter.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int, or (height, width)
            batch=1,
            simplify=True,    # onnxsim over the ONNX intermediate
            output_path=None, # None writes weights/<stem>.tflite
            verbose=False,    # True streams the onnx2tf log
        )

        # dynamic=True raises ValueError: the converter needs static shapes.
        # half=True and int8=True are rejected before tracing.
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Bare LiteRT
      language: python
      code: |
        import json

        import numpy as np
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")
        interpreter.allocate_tensors()
        detail = interpreter.get_input_details()[0]
        print(detail["shape"], detail["dtype"])   # NHWC, not NCHW

        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"], np.float32))
        interpreter.invoke()
        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Class names, task and input size live in the sidecar.
        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Preprocessing, the NCHW-to-NHWC transpose and postprocessing are yours.
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

<code-tabs name="install" />

The extra pulls `onnx2tf` for the conversion and `ai-edge-litert` for running the
result, both behind a Python 3.12 marker. On an older interpreter the export
raises an `ImportError` that names the version requirement rather than failing
inside the converter.

`libreyolo[litert]` installs exactly the same thing. The format string `litert` is
an alias for `tflite`, and the output file is a `.tflite` either way.

## Export

<code-tabs name="export" />

The family and task are checked before anything else happens, so an unsupported
combination fails immediately with the specific converter or runtime error that
kept it out, not a generic message. The conversion itself is a subprocess call to
`onnx2tf` in `flatbuffer_direct` mode over a static ONNX intermediate.

Metadata is a sidecar. `weights/LibreYOLO9t.tflite.json` carries the family, task,
class names, input size and pose schema; the FlatBuffer itself has no LibreYOLO
metadata field, so the two files travel together.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` dispatches on the `.tflite` suffix and returns the same `Results`
object as the checkpoint. The backend reads the sidecar, transposes the NCHW blob
to NHWC when the interpreter asks for a channels-last input, applies the
interpreter's quantization scale and zero point where present, and transposes
outputs back into the layout LibreYOLO's postprocessing expects.

The second snippet is the bare-runtime path. Preprocessing, the layout transpose,
decoding, NMS and coordinate rescaling all become yours there, and the layout
detail is the one most likely to be missed: onnx2tf emits channels-last inputs,
so a blob shaped `(1, 3, 640, 640)` will not bind.

## Constraints

Static shapes only. `dynamic=True` raises `ValueError` before tracing, and the
export canvas is fixed at whatever `imgsz` resolved to.

FP32 only. `half=True` and `int8=True` are both rejected during validation, so
quantized deployment is not reachable from this exporter today.

Coverage is narrower here than for the graph formats, and it is decided by
measurement rather than by family. Validated combinations include YOLO9, YOLOX
and YOLO-NAS detection, PIDNet semantic segmentation, the four CNN classification
families, DINOv2 and SigLIP2 embedding, SigLIP2 classification, TEED and DexiNed
edge, and Real-ESRGAN and SwinIR restoration. SwinIR carries an extra caveat:
parity holds when the source dimensions match the export canvas exactly, and
smaller sources are padded to the canvas before the transformer runs, which can
diverge from native variable-size inference.

The blocked entries name the exact failure, which is worth reading before
attempting a workaround. A few examples: RF-DETR detection converts at its native
384 canvas but LiteRT cannot allocate it because `STRIDED_SLICE` receives an input
above its supported 5-D rank; PicoDet is rejected because a `RESHAPE` maps 19,200
input elements to 9,600 output elements; D-FINE crashes the converter in
`GatherElements` shape handling; RTMDet exports and reloads with raw parity intact
but public boxes fall to 0.911 IoU with 29.9 px of coordinate drift.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination, including
the reason string behind a block:

<code-tabs name="support" />
