---
title: MNN
seo_title: "Export to MNN from LibreYOLO"
description: "Export a LibreYOLO detector to MNN through ONNX and mnnconvert: a fixed NCHW shape, FP32 on CPU, and a metadata sidecar the runtime contract requires."
lead: "MNN is Alibaba's lightweight inference engine. LibreYOLO exports a static ONNX graph, converts it with the mnnconvert tool shipped by the MNN package, and writes a JSON sidecar recording the input and output names, the fixed input shape and the class names."
keywords:
  - yolo mnn export
  - mnnconvert
  - mnn inference
  - mobile detector inference
  - fixed nchw shape
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="mnn")'
    mono: true
  - label: Writes
    value: "One .mnn file plus a .mnn.json metadata sidecar"
  - label: Extra
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t.mnn")'
    mono: true
  - label: Shapes
    value: "Fixed NCHW. dynamic=True is rejected."
  - label: Precision
    value: "FP32 only, CPU only."
  - label: Tasks
    value: "Detection only in this version"
verification: "Read from libreyolo/export/mnn.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/mnn.py and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        # The extra includes libreyolo[onnx]: MNN converts from an ONNX intermediate.
        pip install "libreyolo[mnn]"
    - label: Confirm the converter is on the path
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes weights/LibreYOLO9t.mnn and weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int, or (height, width)
            batch=1,          # baked into the artifact
            simplify=True,    # onnxsim over the ONNX intermediate
            output_path=None, # None writes weights/<stem>.mnn
            verbose=False,    # True streams the mnnconvert log
        )

        # dynamic=True raises ValueError. half=True and int8=True are rejected.
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Bare MNN
      language: python
      code: |
        import json

        import MNN
        import numpy as np

        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))
        print(meta["mnn_input_names"], meta["mnn_output_names"], meta["mnn_input_shape"])

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

        # Preprocessing and postprocessing are yours on this path.
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

<code-tabs name="install" />

The extra includes `libreyolo[onnx]`, because the conversion runs over an ONNX
intermediate. It also brings the `mnnconvert` executable, which the exporter looks
for next to the active Python interpreter first and on `PATH` second. A missing
converter raises an `ImportError` naming the install command rather than failing
mid-conversion.

## Export

<code-tabs name="export" />

Before handing the graph over, the exporter reads the ONNX input contract and
refuses anything it cannot express: more than one image input, or an input shape
with a symbolic dimension. MNN in this version requires a fully fixed NCHW shape,
and `batch` is baked into the artifact rather than negotiated at load time.

The sidecar is not optional bookkeeping. `weights/LibreYOLO9t.mnn.json` records
the input and output names, the fixed input shape, the batch, the class names, the
MNN version used, and the backend the artifact was built for, and the runtime
validates every one of those fields on load.

On Windows, MNN 3.6.1 sometimes completes the conversion and then terminates
during process teardown with an access violation or a fail-fast status. The
exporter recognizes those specific exit codes and treats the conversion as
successful when the output file is present.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` dispatches on the `.mnn` suffix and returns the same `Results` object
as the checkpoint. The load is strict by design: the sidecar has to declare
`format=mnn`, `mnn_backend=cpu`, `dynamic=false`, `precision=fp32`, a size, a
detection task, a fixed positive NCHW shape that agrees with the recorded image
size, and class names covering every index from 0 to `nc - 1`. Any mismatch raises
rather than guessing.

Prediction at a different `imgsz` than the artifact was built for raises too, and
`device` is ignored with a warning, because MNN exports run on CPU here.

The second snippet is the bare-runtime path. Preprocessing, decoding, NMS and
coordinate rescaling become yours there, and the input and output names come from
the sidecar because MNN's module loader wants them explicitly.

## Constraints

Detection only. The backend refuses any other task on load, and the export side
matches: outside the recorded combinations, preflight raises with "MNN v1 has no
implemented runtime contract for this family and task."

FP32, CPU, fixed shape. `dynamic=True` raises `ValueError`, and `half=True` and
`int8=True` are rejected during validation.

Validated detection families are YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC, RT-DETR,
RT-DETRv2, RT-DETRv4, D-FINE, DEIM and YOLO-NAS, each covered by conversion, a
fresh artifact reload, MNN CPU execution, metadata checks and matched post-NMS
detection parity against the PyTorch model. DEIMv2 converts, reloads, executes and
preserves post-NMS detections, but its intermediate ONNX route has incomplete
query-level score parity, so it is recorded as available rather than validated.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination:

<code-tabs name="support" />
