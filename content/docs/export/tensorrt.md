---
title: TensorRT
seo_title: "Export to TensorRT from LibreYOLO"
description: "Build a TensorRT engine from a LibreYOLO model: the ONNX intermediate, FP16 and INT8 builds, dynamic batch profiles, and engine portability limits."
lead: "TensorRT compiles a graph into an engine tuned for one GPU. LibreYOLO exports an ONNX intermediate first, parses it with TensorRT's ONNX parser, builds the engine, and writes the model metadata beside it as a JSON sidecar."
keywords:
  - yolo tensorrt export
  - tensorrt engine
  - trt fp16
  - tensorrt int8 calibration
  - optimization profile
  - dynamic batch tensorrt
  - hardware compatibility level
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="tensorrt")'
    mono: true
  - label: Writes
    value: "One .engine file plus a .engine.json metadata sidecar"
  - label: Extra
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t.engine")'
    mono: true
  - label: Shapes
    value: "Static by default; dynamic=True adds a batch-axis optimization profile"
  - label: Precision
    value: "FP32, FP16 (half=True), INT8 (int8=True with data=)"
  - label: Requires
    value: "An NVIDIA GPU at build time and at run time. Engines do not move between GPU architectures."
verification: "Read from libreyolo/export/tensorrt.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/tensorrt.py and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        # The engine is built from an ONNX intermediate, so both extras are needed.
        pip install "libreyolo[onnx,tensorrt]"
    - label: Confirm the toolchain before building
      language: bash
      code: |
        python -c "import tensorrt, torch; print(tensorrt.__version__, torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes weights/LibreYOLO9t_fp16.engine and weights/LibreYOLO9t_fp16.engine.json
        path = model.export(format="tensorrt", half=True)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Arguments
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # required when int8=True
            dynamic=False,
            workspace=4.0,                  # GiB of build-time scratch
            min_batch=1,                    # dynamic profile bounds
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # or "ampere_plus"
            gpu_device=0,                   # build device on a multi-GPU host
            verbose=False,
        )
  dynamic:
    - label: Dynamic batch engine
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The ONNX intermediate needs the dynamic batch axis for the profile
        # to have anything to bind to.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 with calibration data
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # required: there is no default for this format
            fraction=1.0,
        )
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        results = model.predict(SAMPLE_IMAGE)
        print(results.boxes.xyxy[:3])
    - label: Bare TensorRT
      language: python
      code: |
        import json

        import tensorrt as trt

        path = "weights/LibreYOLO9t_fp16.engine"
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Class names, task and input size live in the sidecar, not the engine.
        # Buffer allocation, preprocessing and postprocessing are yours here.
        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Check one family and task before building
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

Both the build and the run need an NVIDIA GPU with a working CUDA stack. There
is no CPU fallback for this format.

<code-tabs name="install" />

The `tensorrt` extra pins `tensorrt-cu12` and `pycuda`, and the marker drops both
on macOS. On a Jetson, do not use that extra: it pins a CUDA 12 build against a
CUDA 13 platform. Use the TensorRT that JetPack installs instead, as described on
[NVIDIA Jetson](/docs/export/jetson).

## Export

<code-tabs name="export" />

The export runs in two steps. Step one writes an ONNX intermediate to a temporary
path, step two parses it and builds the engine, and the intermediate is removed
afterwards. `workspace` is build-time scratch memory in GiB; a larger value lets
the builder try more kernels and does not affect inference memory.

The metadata sidecar is written next to the engine as `<engine>.json` and records
the precision the build actually realized. When the GPU lacks fast FP16 or fast
INT8 the builder warns and falls back, and the sidecar reports the precision that
came out rather than the one that was asked for.

Under FP16, a ViT backbone in the graph is detected and its float layers are
pinned to FP32. DINOv2-style backbones overflow in FP16 and produce NaN, so the
build sets `OBEY_PRECISION_CONSTRAINTS` and reports `FP16 (FP32 ViT backbone)`.
The pass is a no-op on CNN backbones.

### Dynamic batch

<code-tabs name="dynamic" />

`dynamic=True` adds one optimization profile spanning `min_batch` to `max_batch`,
optimized at `opt_batch`, and records those three values in the sidecar. The
profile is only added when the ONNX intermediate actually carries a dynamic batch
dimension; otherwise the build logs that it is using static optimization and
continues.

### INT8

<code-tabs name="int8" />

INT8 uses TensorRT's entropy calibrator over a LibreYOLO calibration loader, and
`data` is mandatory: this format has no eight-image fallback. Calibration needs
`cuda-python` or `pycuda` for the device buffer. The calibration cache is keyed
on a hash of the ONNX bytes, so scales from one model are never reused for
another that happens to write to the same output path.

`half=True` and `int8=True` together warn and build INT8, which keeps an FP16
fallback for layers TensorRT cannot quantize.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` dispatches on the `.engine` suffix, reads the sidecar for class
names, task and pose schema, and returns the same `Results` object as the
checkpoint. It raises immediately when no CUDA device is present.

The second snippet is the bare-runtime path. Host and device buffer allocation,
preprocessing, decoding, NMS and coordinate rescaling all become yours, and the
engine itself carries no class names, so the sidecar has to travel with it.

## Constraints

A serialized engine is tied to the GPU architecture, the driver stack and the
TensorRT version that built it. An engine built on a workstation will not load on
a different architecture, which is why the build step runs on the deployment
machine. `hardware_compatibility="ampere_plus"` trades some performance for
portability across Ampere and newer. The `"same_compute_capability"` value maps to
`NONE` on TensorRT builds that do not expose that level, and in that case the
export warns rather than claiming portability it did not apply.

Only the batch axis is profiled. A build with dynamic spatial dimensions is not
part of this contract, which is why FCOS is blocked: it needs dynamic padded
height and width to preserve its 800 by 1333 aspect transform.

Blocked before tracing: YOLO9 segmentation, RTMDet-Ins segmentation, SSD,
Faster R-CNN and RetinaNet detection, and BiRefNet or FeyNobg matting, where
TensorRT 10.16 reaches the shared ONNX `DeformConv` node and cannot parse it
because `ModulatedDeformConv2d` is absent from the plugin registry.

Where a combination is neither validated nor blocked, the converter path is
available and the project has not recorded TensorRT runtime parity for it. That
is a statement about evidence, not about whether the build succeeds.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination:

<code-tabs name="support" />
