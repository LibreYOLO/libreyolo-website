---
title: Install
seo_title: "Install LibreYOLO"
description: "Install LibreYOLO from PyPI, pick the optional extras a model family or export target needs, and confirm PyTorch sees your GPU."
lead: "LibreYOLO is published on PyPI as libreyolo. The base package covers prediction, training, validation and the model families that need nothing beyond PyTorch; optional extras add the rest."
keywords: [libreyolo install, pip install libreyolo, libreyolo extras, libreyolo cuda, libreyolo gpu, libreyolo requirements]
last_verified: "1.5.0"
meta:
  - label: Package
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 or newer
  - label: Code license
    value: MIT
  - label: Core dependency
    value: PyTorch 2.4 or newer
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: With extras
      language: bash
      code: |
        # Comma-separate to combine several in one install.
        pip install "libreyolo[rfdetr,onnx]"
    - label: Everything
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: From source
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, every visible GPU, and which
        # optional packages are installed.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Model inventory
      language: bash
      code: |
        # Every registered family with its tasks, sizes and input
        # resolutions. Families whose extra is missing are listed with
        # the pip command that enables them.
        libreyolo models
---

## Install

<code-tabs name="install" />

Python 3.10 or newer is required. The base install pulls PyTorch, torchvision,
NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools, typer, click,
safetensors and SciPy, so YOLOv9 and the other families that need nothing more
work straight after `pip install libreyolo`.

A clone checks out `release`, the stable branch whose code matches these docs.
The integration branch, carrying unreleased work, is `dev`.

## Optional extras

An extra is a bracketed name that adds the dependencies one model family or one
export target needs. Nothing else changes: the API is the same whether or not
an extra is present.

### Model families

| Extra | Adds |
|---|---|
| `rfdetr` | `transformers`, which supplies the RF-DETR backbone |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, which supplies MiDaS's ViT-L/16 and EfficientNet-Lite3 encoders |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate`, and `bitsandbytes` off macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` and `ftfy`, needed by the vendored CLIP text tokenizer |
| `siglip2` | `sentencepiece`, needed by the multilingual SigLIP 2 tokenizer |
| `gaze` | `gdown`, which turns on auto-download of the L2CS checkpoint |
| `rtdetr` | Nothing. RT-DETR needs no extra dependency; the name is kept stable |

### Export and runtimes

| Extra | Adds |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 and `pycuda`, off macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, macOS only |
| `tflite`, alias `litert` | `libreyolo[onnx]` plus `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` and `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` plus `MNN` |
| `ncnn` | `pnnx` and `ncnn` |
| `paddle` | `libreyolo[onnx]` plus `paddlepaddle` 2.6.2 and `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` for HTTP and HTTPS V2 inference |

### Training, evaluation and logging

| Extra | Adds |
|---|---|
| `lora` | `libreyolo[rfdetr]` plus `peft`, for `lora=True` fine-tuning |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, the C++ COCO evaluation backend |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, alias `dvc` | `dvclive` |

`fast-eval` is opt-in rather than a hard dependency so that a platform without a
prebuilt wheel cannot break a plain install. When the package is absent, COCO
evaluation falls back to pycocotools and the run continues.

### Tooling

| Extra | Adds |
|---|---|
| `stream` | `yt-dlp`, needed only to resolve YouTube page URLs |
| `tracking` | Nothing. Every tracking dependency is already a core dependency |
| `label` | `libreyolo[sam]`, which enables click-to-mask assist in `libreyolo label` |
| `hub-kernels` | `kernels`, the optional loader for compiled Hub kernels. See [kernels](/docs/reference/kernels), which notes that installing it can shift RF-DETR predictions at float tolerance |
| `clip-convert` | `libreyolo[clip]` plus `open_clip_torch`, for weight conversion and parity checks |
| `siglip2-convert` | `libreyolo[siglip2]` plus `transformers`, for the same reason |

Webcams, RTSP, RTMP, TCP, UDP, HLS and local multi-stream lists need no extra.
Only YouTube page URLs do.

### The aggregate extra

`libreyolo[all]` installs the model, export, tracking and logging extras in one
command. Some are deliberately outside it. `neptune` is excluded because stable
`neptune-scale` requires protobuf below 7 while the TFLite path requires
protobuf 7. `executorch` is excluded because ExecuTorch constrains which PyTorch
version it pairs with, and `coreai` because `coreai-torch` pins PyTorch to
2.11.x and would drag the whole environment onto that version. `fast-eval`,
`hub-kernels`, `clip-convert` and `siglip2-convert` are also left out. Install
any of them by name.

## Platform constraints

Three extras are platform-scoped by their dependency markers, so the install
succeeds everywhere and simply installs less where a wheel does not exist.

| Extra | Constraint |
|---|---|
| `coreai` | macOS only. The Core AI toolchain neither converts nor runs elsewhere |
| `tensorrt` | Skipped on macOS, which has no CUDA |
| `tflite`, `litert` | `onnx2tf` and `ai-edge-litert` require Python 3.12 or newer |

`sensenova` skips `bitsandbytes` on macOS, where no wheel is published; the rest
of the extra installs normally.

If disk is the constraint, most of it is PyTorch, and most of PyTorch is the
CUDA payload its default wheel bundles. A CPU-only wheel removes that without
giving anything up. For ONNX detection on a machine that should carry no torch
at all, see the [lightweight install](/docs/lightweight-install).

## GPU and CUDA

Device selection happens when a model is constructed. The default,
`device="auto"`, uses CUDA when `torch.cuda.is_available()` is true, then Metal
Performance Shaders when `torch.backends.mps.is_available()` is true, and CPU
otherwise. Nothing else in the library inspects the hardware, so if PyTorch
cannot see a GPU, neither can LibreYOLO.

To pin the device instead, pass `device` to the model or to `predict`, `train`,
`val` and `export`. It accepts `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, a bare
integer such as `0`, or a digit string such as `"0"`; the last two are expanded
to `cuda:<n>`.

Start with `libreyolo checks`, which prints the Torch version, the CUDA and
cuDNN versions Torch was built against, and every visible GPU with its memory.
When it reports no CUDA on a machine that has an NVIDIA card, the PyTorch wheel
pip resolved is a CPU build. Install a CUDA build from the PyTorch index first,
then install LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

That is the same index the repository pins for its own uv-managed environment on
Linux and Windows. It needs NVIDIA driver 555 or newer, which is the CUDA 12.8
runtime requirement. macOS keeps the PyPI wheel, since the PyTorch download host
publishes no Darwin builds.

## Check the install

<code-tabs name="verify" />

`libreyolo models` is the fastest way to see whether an extra took effect: a
family whose dependency is missing is printed with the exact pip command that
enables it. Both commands also accept `--json`, which prints the same data as a
machine-readable object on stdout.
