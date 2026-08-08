---
title: NVIDIA Jetson
seo_title: "Install LibreYOLO and PyTorch on NVIDIA Jetson"
description: "Install LibreYOLO on an NVIDIA Jetson: the four CUDA libraries JetPack leaves out, the --no-deps step PyTorch needs, and measured Orin Nano numbers."
lead: "NVIDIA Jetson boards run LibreYOLO on the standard aarch64 PyTorch wheels. No Jetson-specific torch build is involved, but JetPack omits four libraries that torch links against, and the install has to supply them."
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - install PyTorch on Jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - TensorRT on Jetson
  - aarch64 wheels
last_verified: "1.4.0"
meta:
  - label: Board
    value: "Jetson Orin Nano Super Developer Kit, 8 GB, GPU compute capability 8.7"
  - label: Platform
    value: "JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64"
  - label: Stack tested
    value: "libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv 5.0.0, numpy 2.5.1, on 2026-07-27"
  - label: Missing from JetPack
    value: "nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13, nvidia-nvshmem-cu13"
    mono: true
  - label: Benchmarked
    value: "223 verified runs on this board, 58 models across 12 families, in PyTorch, ONNX Runtime and TensorRT"
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: https://www.visionanalysis.org/hardware/jetson_orin
  - label: Tracked in
    value: "The Jetson half of issue 648"
    links:
      - label: issue 648
        href: https://github.com/LibreYOLO/libreyolo/issues/648
verification: "Install recipe and expected output taken from the install run of 2026-07-27 on a Jetson Orin Nano Super. Latency and accuracy rows come from the verified-results snapshot behind visionanalysis.org, filtered to hardware jetson_orin, measured June 2026 on libreyolo 1.2.0.dev0. Export and loader behavior read from libreyolo/export/exporter.py, libreyolo/export/tensorrt.py and libreyolo/models/__init__.py."
snippets:
  prep:
    - label: System packages and a virtual environment
      language: bash
      code: |
        # JetPack does not preinstall pip or the venv module.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: PyTorch, from the CUDA 13 wheel index
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: The four libraries JetPack does not ship
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: If pip demands cuda-toolkit 13.0.3, install with --no-deps
      language: bash
      code: |
        # --no-deps means torch's Python dependencies are named by hand too.
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Name the next missing library instead of guessing
      language: bash
      code: |
        ldd "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Everything still missing across all of torch's libraries, in one pass:
        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so 2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: Install LibreYOLO after torch, not before
      language: bash
      code: |
        # torch is already satisfied, so pip leaves the CUDA build in place.
        pip install libreyolo

        # The ONNX extra is only needed to export. A TensorRT export runs
        # through ONNX, so add it before the export section below.
        pip install "libreyolo[onnx]"
  verify:
    - label: Versions and device
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: Then run a real kernel
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9

        # Downloads the checkpoint on first use.
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict("bus.jpg")
        print(result.boxes)
    - label: CLI
      language: bash
      code: |
        libreyolo predict --source bus.jpg --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, LibreYOLO

        # Writes libreyolo9s.onnx, then builds libreyolo9s.engine from it.
        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt", half=True)

        # The engine loads back through the same entry point.
        result = LibreYOLO("libreyolo9s.engine").predict("bus.jpg")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Power mode and clocks
      language: bash
      code: |
        sudo nvpmodel -q      # which modes this board exposes, and the active one
        sudo nvpmodel -m 0    # highest mode on the board tested here
        sudo jetson_clocks

        tegrastats            # live load; nvidia-smi is limited on Tegra
---

## What this page records

This page records one configuration that was verified end to end, not a support
matrix. The board was a Jetson Orin Nano Super Developer Kit with 8 GB of memory
running JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13, Python 3.12.3), and the
stack that came up on it was `libreyolo 1.4.0` with `torch 2.13.0+cu130`, OpenCV
5.0.0 and NumPy 2.5.1. `torch.cuda.is_available()` returned `True` and the GPU
reported itself as `Orin`.

Other JetPack releases, other Jetson boards and other CUDA versions were not
tested. The recipe below is the one that worked on that combination.

Two things about it run against what most Jetson guides say. The wheels are the
ordinary aarch64 builds published for CUDA 13, so no Jetson-specific torch build
is needed. And JetPack does not ship four libraries that those wheels link
against, so `import torch` fails one library at a time until all four are
installed.

## Install

JetPack images arrive without pip and without the `venv` module, so both come
first.

<code-tabs name="prep" />

An 8 GB board is tight for larger checkpoints. Adding swap on the NVMe before
loading them avoids an out-of-memory kill mid-run.

Then PyTorch. The CUDA 13 index carries the aarch64 wheels; the extra index
supplies the pure-Python dependencies from PyPI.

<code-tabs name="torch" />

The four `nvidia-*-cu13` wheels are the part that is easy to miss. JetPack
provides the GPU driver, not cuDNN, NCCL, cuSPARSELt or NVSHMEM, and torch
refuses to import without them. Installing all four at once is faster than
discovering them one exception at a time.

The third snippet covers a specific failure: torch's dependency metadata for the
CUDA 13 build asks for `cuda-toolkit==13.0.3`, which has no aarch64 wheel on
PyPI, so resolution fails before anything downloads. `--no-deps` skips the
resolver, which means every dependency has to be named on the command line.

LibreYOLO goes in last. Installing it first lets pip choose its own torch, which
on this platform is not the CUDA build.

<code-tabs name="install" />

Every remaining dependency resolves to a prebuilt aarch64 wheel, including
OpenCV, NumPy, SciPy, pycocotools and safetensors. Nothing compiles from source.

## Check that CUDA works

<code-tabs name="verify" />

The second snippet matters as much as the first. A wheel built for the wrong GPU
architecture still reports `torch.cuda.is_available() == True` and then fails on
the first real operation with `CUDA error: no kernel image is available for
execution on the device`. A matrix multiply on the device is the check that
catches it.

## Run a prediction

<code-tabs name="predict" />

`predict` returns the same `Results` object as on any other platform, so model
pages apply unchanged.

## Export to TensorRT

On this board, TensorRT was faster than both PyTorch and ONNX Runtime for all 55
models that were measured in every runtime.

<code-tabs name="export" />

`format="tensorrt"` writes an ONNX graph first and builds the engine from it, so
the `onnx` extra has to be installed. `LibreYOLO()` dispatches on the file
suffix, so a `.engine` file loads through the same call as a `.pt` checkpoint.

Do not use the `tensorrt` pip extra on a Jetson. It pins `tensorrt-cu12`, a CUDA
12 build, against a CUDA 13 platform. Use the TensorRT that JetPack installs
instead. If `import tensorrt` fails inside the virtual environment while it
works outside, recreate the environment with `--system-site-packages` so the
system module is visible.

Serialized TensorRT engines are tied to the device, the GPU architecture and the
TensorRT version that built them. An engine built on a workstation will not load
on a Jetson, so the build step runs on the board.

## Measured on this board

Latency per image, batch size 1, end to end including preprocessing and
postprocessing, on COCO val2017 (500-image subset) at `conf=0.001` and
`max_det=300`. Five models out of the 58 measured:

| Model | Input (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

The mAP column is the TensorRT FP16 run's own score. Across the 55 models
measured in all four runtimes, the largest gap between the PyTorch FP32 score and
the TensorRT FP16 score was 0.59 points, on DEIMv2-X. The runtimes differ in
speed, not in accuracy.

TensorRT FP32 was faster than both PyTorch and ONNX Runtime for all 55 of those
models. TensorRT FP16 was faster than PyTorch FP32 for all 55 as well, by 1.68x
to 6.22x, with a median of 3.39x. ONNX Runtime is the one that varies: it was
slower than PyTorch on 23 of the 55, the RT-DETR-r18 row among them.

Conditions behind every number: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, driver 595.78, ONNX Runtime 1.24.0, measured June 2026.
Latency on a Jetson also depends on the active power mode, which the benchmark
records do not carry.

<code-tabs name="power" />

All 223 runs, including the other 53 models and the full accuracy columns, are
published on
[the Jetson Orin page at Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Troubleshooting

### import torch fails naming a shared library

One of the four libraries above is missing. Rather than guessing which, read it
off the binary:

<code-tabs name="ldd" />

Each missing entry maps to one wheel:

| Missing library | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch warns that no build supports this GPU

The first CUDA call on the working configuration prints this:

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

The warning is cosmetic on this board. The wheel carries `sm_80` kernels and the
Orin executes them. The same warning appeared on the earlier wheel from that
index, the one that produced every benchmark row above. Confirm with the matrix
multiply from the CUDA check rather than trusting or distrusting the message.

### CUDA error: no kernel image is available for execution on the device

The installed wheel was built for a different GPU architecture. This is what
happens with wheels from NVIDIA's `sbsa` index, which target server ARM GPUs
rather than Jetson silicon. Reinstall from the CUDA 13 index in the install
section.

### pip cannot find cuda-toolkit 13.0.3

There is no aarch64 wheel for it. Use the `--no-deps` form in the install
section and name torch's dependencies explicitly.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

The aarch64 torch wheel links NVIDIA Performance Libraries for CPU math. Install
them and put them on the library path:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

That index is fine for these two CPU libraries. Its torch builds are the ones
that produce the "no kernel image" failure above.

### Wheel sources that do not fit JetPack 7.2

| Source | Result on the Orin Nano Super |
|---|---|
| `pypi.jetson-ai-lab.io/sbsa/cu130` torch | Built for server ARM GPUs. Imports, reports CUDA available, then fails with "no kernel image is available for execution on the device". |
| `pypi.jetson-ai-lab.io/jp6/*` torch | CUDA 12 and Python 3.10 builds. They do not install on this image's Python 3.12. |
| JetPack 6 PyTorch containers | CUDA initialization fails with error 801 on a JetPack 7 host. |
| Building torch from source | Works, but takes hours on an 8 GB board and is unnecessary once the CUDA 13 wheels are installed. |

## DeepStream

For a full video pipeline rather than a Python loop, export with
`deepstream=True` and run the graph through `nvinfer`. That path has its own
page, including the generated `nvinfer` config, the bounding-box parser build and
the known traps: [DeepStream](/docs/export/deepstream).

The DeepStream pipeline itself was validated on an x86 discrete GPU, not on a
Jetson. The export contract does not depend on the architecture, but the pipeline
run on aarch64 is still outstanding.

## Not verified

- JetPack releases other than 7.2, and L4T releases other than R39.2.
- Jetson boards other than the Orin Nano Super 8 GB.
- Training on the board. Inference and export were exercised; a training run was
  not.
- INT8 engines. Only FP32 and FP16 rows exist for this board.
- Batch sizes above 1. Every measurement above is batch 1.
