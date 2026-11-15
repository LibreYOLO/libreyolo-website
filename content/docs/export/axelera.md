---
title: Axelera Metis
seo_title: "Export LibreYOLO models to Axelera Metis"
description: "Compile a LibreYOLO model to an INT8 .axm for the Axelera Metis AIPU: the SDK you install yourself, the export arguments, the parity check, and which families and sizes compile."
lead: "format=\"axelera\" compiles a LibreYOLO model to an INT8 .axm for Axelera's Metis AIPU, and the exported directory loads back through LibreYOLO() with the same predict, val and track calls as the PyTorch model. New in 1.8.0, and marked preview."
keywords:
  - libreyolo axelera
  - axelera metis
  - metis aipu
  - voyager sdk
  - axelera-devkit
  - axelera-rt
  - int8 npu export
  - axm model
last_verified: "1.8.0"
meta:
  - label: Format
    value: 'export(format="axelera")'
    mono: true
  - label: Precision
    value: "INT8 only. half=True, quantize=16 and quantize=32 raise."
  - label: Compile host
    value: "Linux x86_64 (Ubuntu 22.04 or 24.04, or WSL2), Python 3.10 to 3.13. No Metis device needed to compile."
  - label: SDK
    value: "axelera-devkit and axelera-rt 1.9.0 from Axelera's own package index, not a LibreYOLO extra"
  - label: Output
    value: "weights/<stem>_axelera_model/"
    mono: true
  - label: Status
    value: "Preview. 23 families compiled and measured on one Metis M.2 Max with SDK 1.9.0."
verification: "Read from docs/axelera.md, libreyolo/export/axelera.py, libreyolo/export/support.py and libreyolo/cli/commands/export.py on the dev branch at the 1.8.0 release candidate. The accuracy figures are the library's own paired INT8 vs FP32 measurements; this site did not rerun them."
snippets:
  install:
    - label: Dedicated environment
      language: bash
      code: |
        python -m venv ~/venvs/axelera && source ~/venvs/axelera/bin/activate
        pip install "libreyolo[onnx]"   # the onnx extra is needed to compile

        # Install the SDK last: its wheels pin packages LibreYOLO also uses.
        # Compile (export):
        pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-devkit==1.9.0
        # Run (.axm inference):
        pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt==1.9.0
    - label: Let LibreYOLO install it
      language: bash
      code: |
        # Without this, a missing SDK fails with the pip command above.
        export LIBREYOLO_AUTOINSTALL=1
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        path = model.export(format="axelera")   # weights/LibreYOLO9t_axelera_model

        # Calibrate on your own domain and write a parity report.
        path = model.export(
            format="axelera",
            data="my_dataset.yaml",
            fraction=0.5,
            verify=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=axelera
        libreyolo export model=LibreYOLO9t.pt format=axelera verify=true
  run:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        npu = LibreYOLO("weights/LibreYOLO9t_axelera_model")   # or the .axm inside it
        results = npu(SAMPLE_IMAGE, conf=0.25)
        metrics = npu.val(data="coco128.yaml")
        for r in npu.track("video.mp4"):
            print(r.track_id)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=weights/LibreYOLO9t_axelera_model source=image.jpg
        libreyolo val model=weights/LibreYOLO9t_axelera_model data=coco128.yaml
        libreyolo track model=weights/LibreYOLO9t_axelera_model source=video.mp4
---

## Install

The Axelera SDK is proprietary software under Axelera's EULA, distributed from
Axelera's own package index. It is not on PyPI and not a LibreYOLO extra, and
LibreYOLO never bundles it. `axelera-devkit` compiles; `axelera-rt` runs the
compiled model and is the only one a deployment machine needs.

<code-tabs name="install" />

Use a dedicated virtual environment. The SDK 1.9.0 wheels pin several packages
LibreYOLO also uses, and pip downgrades them on install:

| Package | Pinned to |
|---|---|
| `torch` | `>=2.7,<2.15` |
| `typer` | `>=0.12,<0.13` |
| `click` | `==8.1.8` |
| `rich` | `<14` |
| `onnx` | `<1.19` |
| `opencv-python` | `<5` |
| `pillow` | `<13` |

LibreYOLO works with all of these. Installing other LibreYOLO extras after the
SDK can upgrade them again, so install the SDK last. When a package is missing,
export or load fails with the exact `pip install` command; set
`LIBREYOLO_AUTOINSTALL=1` to let LibreYOLO run it for you, which it logs.

Running needs a Metis device, Axelera's `metis-dkms` kernel driver (1.7.0 or
newer for SDK 1.9) and `axelera-rt`. `axdevice` lists each device with its
firmware, and `libreyolo checks` lists the Axelera packages and the Metis
devices it sees.

## Export

<code-tabs name="export" />

| Argument | Axelera behavior |
|---|---|
| `imgsz` | Fixed in the compiled model; a multiple of 32. Default: the checkpoint's size |
| `data` | Calibration dataset YAML. Default `coco128.yaml` for detect and pose |
| `fraction` | Share of the calibration split to use. 100 to 400 images are recommended; fewer than 100 logs a warning |
| `batch` | Calibration loader batch size only. The compiled model takes one image at a time |
| `device` | Torch device used to trace the model, not the Metis device |
| `quantize`, `int8` | INT8 is always on. `quantize=16`, `quantize=32` and `half=True` raise |
| `dynamic` | Always static; `dynamic=True` raises |
| `opset` | Opset of the intermediate ONNX graph, 14 to 17 (default 17) |
| `nms` | Not supported: NMS runs on the host |
| `output_path` | Default `weights/<checkpoint stem>_axelera_model` |
| `verify` | Writes `<stem>.parity.json`, comparing the quantized model with FP32, and with the device when one is present |
| `compiler_config` | Python only: a dict of Axelera `CompilerConfig` fields. Unknown fields and the ones LibreYOLO owns raise |

Calibration uses the model's own preprocessing. For a fine-tuned model,
calibrate on images from its own domain. Compiling takes minutes to tens of
minutes and several GB of RAM.

```text
weights/LibreYOLO9t_axelera_model/
    LibreYOLO9t.axm               compiled model
    compiler_config_final.toml    compiler settings used for the build
    metadata.yaml                 LibreYOLO metadata
    LibreYOLO9t.parity.json       only with verify=True
```

The directory is named after the checkpoint file. Pretrained YOLO-NAS weights
are converted on first use, so their exports land in
`weights/LibreYOLONASs-converted_axelera_model`. Re-exporting to the same path
replaces the directory only after the new build succeeded, and no ONNX file or
compiler build files are left behind.

## Run the artifact

<code-tabs name="run" />

The network runs on the Metis AIPU; preprocessing, box decode and NMS run on
the host, so box coordinates and scores never pass through an 8-bit tensor.
Results are the same `Results` objects as from the PyTorch model. The input size
is fixed by the export. `device="auto"`, `"axelera"`, `"metis"`, `"aipu"` and
`"cpu"` all run on the AIPU; a CUDA device is ignored with a warning. Copy the
directory to any Linux machine with a Metis device and `axelera-rt`.

## Supported families

Every size below compiles, runs on a Metis device and exports. "Within 1.0"
means the INT8 result is within 1.0 point of FP32 on the paired set; a size
with a larger measured drop still exports, and the export logs that drop once
as a warning. Accuracy never blocks an export. Each family and task has its own
`validated` or `available` cell in the export matrix (`libreyolo formats
--family yolonas` prints them), and anything not listed raises before
compiling.

| Family | Task | Sizes | Measured drop |
|---|---|---|---|
| YOLO-NAS | detect | s, m, l | within 1.0 |
| YOLO-NAS | pose | n, s, m, l | n and s larger |
| YOLO-NAS-R | obb | s, m, l | within 1.0; needs calibration data |
| YOLO9 | detect | t, s, m, c | within 1.0 |
| YOLOX | detect | n, t, s, m, l, x | n larger |
| YOLOv2 | detect | t, b | within 1.0 |
| YOLOv3 | detect | t, b, spp | within 1.0 |
| YOLOv4 | detect | t, b | b larger |
| YOLOv7 | detect | b | within 1.0 |
| PP-YOLOE | detect | s, m, l, x | s, m and x larger |
| PicoDet | detect | s, m, l | larger |
| RTMDet | detect | t, s, m, l | larger |
| RTMDet-Ins | segment | t, s, m, l | larger |
| RetinaNet | detect | r50 | larger |
| SSD | detect | 300 | within 1.0 |
| EfficientDet | detect | d0, d1, d2 | larger |
| ResNet | classify | 18, 34, 50, 101 | within 1.0 |
| MobileNetV4 | classify | s, m, l | s and l larger |
| EfficientNetV2 | classify | b0 | larger |
| VGG | classify | 16, 19, 16bn, 19bn | 19bn larger |
| AlexNet | classify | b | within 1.0 |
| FCN | semantic | r50, r101 | within 1.0 |
| DeepLabV3 | semantic | mv3 | larger |
| MiDaS | depth | s | larger |
| QuickSRNet | restore | m2 | larger |
| Real-ESRGAN | restore | x4t | larger |

Some measured deltas, mAP50-95 on 1000 COCO val2017 images, INT8 on the device
against FP32 on the same images: YOLO9-s 0.4502 to 0.4450 (-0.52), YOLO9-t
-0.45, YOLO-NAS-s -0.30, YOLOX-s -0.48, YOLOX-n -1.81 (300 images), PicoDet-m
-2.86, RTMDet-s -6.75. ResNet-50 loses 0.7 points of ImageNetV2 top-1. The
per-size table with every figure is in the library's `docs/axelera.md`.

Blocked for structural reasons: EfficientNetV2 b1 to b3, EfficientDet-d3,
RTMDet-x, RTMDet-Ins-x, DeepLabV3 r50 and r101, MiDaS-l, RetinaNet-r50v2, FCOS,
HRNet, CenterNet, Faster R-CNN, ConvNeXt and ConvNeXt V2, YOLOv1, EC,
TinyFormer, NAFNet, TEED and ZipDepth. RF-DETR and the other transformer
families are out of scope. YOLO9-E2E, YOLO9-P2, EfficientDet-d4, Real-ESRGAN
x4 and x2, PIDNet, DEKR and UNet are not measured yet.

## Licensing

LibreYOLO's code is MIT. `axelera-devkit` and `axelera-rt` are Axelera's
proprietary software, which you install yourself. A compiled model carries the
licence of the weights it came from: YOLO-NAS pretrained weights are
non-commercial and not redistributable, so LibreYOLO hosts no YOLO-NAS `.axm`
files and each user compiles their own. See the
[export matrix](/docs/reference/export-matrix) for how the `axelera` cells are
decided.
