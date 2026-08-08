---
title: ncnn
seo_title: "Export to ncnn from LibreYOLO"
description: "Export a LibreYOLO model to ncnn through PNNX: the param and bin pair, the fixed export canvas, the YOLOX Focus rewrite, and which families convert."
lead: "ncnn is Tencent's CPU inference library for mobile targets. LibreYOLO converts through PNNX, writing a model.ncnn.param graph beside a model.ncnn.bin weight file and a metadata.yaml that carries the family, task and class names."
keywords:
  - yolo ncnn export
  - pnnx
  - model.ncnn.param
  - mobile cpu inference
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="ncnn")'
    mono: true
  - label: Writes
    value: "A directory with model.ncnn.param, model.ncnn.bin and metadata.yaml"
  - label: Extra
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t_ncnn")'
    mono: true
  - label: Shapes
    value: "Fixed. The metadata records dynamic=False regardless of the flag."
  - label: Precision
    value: "FP32 only. half=True and int8=True are rejected."
verification: "Read from libreyolo/export/ncnn.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/ncnn.py and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        # pnnx converts, ncnn runs the result.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes the directory weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int, or (height, width)
            batch=1,
            simplify=True,    # applies to the ONNX fallback path only
            opset=13,         # applies to the ONNX fallback path only
            output_path=None, # None writes weights/<stem>_ncnn
        )

        # half=True and int8=True are rejected during validation.
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        results = model.predict(SAMPLE_IMAGE)
        print(results.boxes.xyxy[:3])
    - label: Bare ncnn
      language: python
      code: |
        import ncnn
        import numpy as np
        import yaml

        directory = "weights/LibreYOLO9t_ncnn"
        net = ncnn.Net()
        net.load_param(f"{directory}/model.ncnn.param")
        net.load_model(f"{directory}/model.ncnn.bin")

        # ncnn takes a single CHW image, not a batch.
        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))
        extractor = net.create_extractor()
        extractor.input("in0", mat_in)
        ret, mat_out = extractor.extract("out0")
        print(ret, np.array(mat_out).shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
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

The extra pulls both halves of the toolchain: `pnnx` performs the conversion and
`ncnn` executes the result. Neither goes through ONNX on the primary path.

## Export

<code-tabs name="export" />

The artifact is a directory. `weights/LibreYOLO9t_ncnn` holds
`model.ncnn.param`, `model.ncnn.bin` and `metadata.yaml`; all three are one
artifact and move together.

Conversion tries PNNX directly from PyTorch first. If that fails it exports a
static ONNX graph to a temporary directory and calls the `pnnx` command line tool
on it, and the export only raises when both paths fail, reporting both errors.
`opset` and `simplify` therefore only affect the fallback.

YOLOX needs one rewrite to convert at all. Its Focus layer uses strided slicing,
which PNNX cannot lower, so the export swaps it for `pixel_unshuffle` and permutes
the following convolution's input channels to compensate for the different channel
ordering. The output is numerically identical, and the original weights are
restored after the export.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` recognizes any directory holding `model.ncnn.param` and
`model.ncnn.bin`, reads `metadata.yaml`, and returns the same `Results` object as
the checkpoint.

The second snippet is the bare-runtime path, and two details differ from every
other format here. ncnn works on a single CHW image rather than a batch, so there
is no leading batch axis. Blob names come from the `.param` file; PNNX writes
`in0` and `out0` by convention, and the backend parses the file rather than
assuming them. Preprocessing, decoding, NMS and coordinate rescaling are yours on
that path.

## Constraints

FP32 on a fixed canvas. `half=True` and `int8=True` are both rejected during
validation, and the exported metadata records `dynamic=False` whatever the flag
said, so no backend assumes an axis the graph does not have.

Every DETR-style family is refused in preflight: `detr`, `deformable_detr`,
`dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`, `rtdetrv2`, `rtdetrv4`,
`rfdetr` and `ec`. The message is the same for all of them, that the model needs
decoder or sampling operations unavailable in ncnn, and it points at ONNX,
OpenVINO, TorchScript or TensorRT instead.

What does convert is broad on the convolutional side: YOLO9 and YOLO9-E2E, YOLOX,
PicoDet, YOLO-NAS detection and pose, the older YOLO1, YOLO3, YOLO4 and YOLO7
detectors, the four CNN classification families, PIDNet semantic segmentation,
FOMO point detection at a fixed 96 by 96, ZipDepth, NAFNet and Real-ESRGAN.

Blocked entries name the concrete failure. Transformer graphs generally leave
unsupported `pnnx.Expression` nodes behind, which produces a network with no
runnable input blob, and that is what stops DINOv2, CLIP, SigLIP2 and SegFormer.
BiRefNet needs torchvision deformable convolution, which PNNX cannot lower.
YOLO2's converted graph terminates the ncnn runtime on Windows with a native
integer divide by zero during output extraction.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination:

<code-tabs name="support" />
