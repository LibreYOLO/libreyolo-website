---
title: Lightweight install
seo_title: "Run LibreYOLO ONNX inference without PyTorch"
description: "Install LibreYOLO with --no-deps and run ONNX detection on numpy alone, no torch on disk. The technique, its limits, and the exact package list."
lead: "LibreYOLO's ONNX inference path is numpy end to end, including decode and NMS. Nothing on it needs PyTorch at runtime, so an install that skips dependency resolution can run detection with torch absent from the machine."
keywords: [torch-free inference, torch free, libreyolo without pytorch, onnx inference no torch, libreyolo lightweight install, pip install no-deps, libreyolo disk space, onnxruntime inference]
last_verified: "1.5.0"
meta:
  - label: Applies to
    value: ONNX detection, seven model families
  - label: Entry point
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Support level
    value: Best effort, not a separate distribution
snippets:
  install:
    - label: Lightweight
      language: bash
      code: |
        # Install the package without its dependency list, then supply the
        # four packages the ONNX detection path actually imports.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: CPU-only torch
      language: bash
      code: |
        # Try this first. It keeps every feature and avoids the CUDA wheel,
        # which is where most of the disk goes.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo.backends.onnx import OnnxBackend

        model = OnnxBackend("libreyolo9t.onnx")
        result = model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")

        # xyxy is a numpy ndarray here, not a torch tensor.
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.boxes.cls)
---

## Why this works

`pip install --no-deps libreyolo` installs the package and skips its dependency
list entirely. Nothing is resolved on your behalf, and you become responsible
for installing what you actually use.

That is only useful if the code path you want genuinely does not need the
dependencies you skipped, and for ONNX detection it does not. The decode,
including non-maximum suppression, is numpy. The preprocessing recipes are
numpy. PyTorch is a training and eager-inference dependency, and on this path
it is never called.

Before this release the import failed anyway: importing anything under
`libreyolo.models` built every model class to populate the checkpoint
auto-detection registry, and those classes are `torch.nn.Module` subclasses.
The preprocessing recipes now live in their own package, `libreyolo.preprocess`,
and the torch import is deferred until something touches a torch attribute, so
the ONNX path imports with torch absent from the machine. That package holds a
numpy-native preprocessor per family: `yolo9`, `yolonas`, `yolox`, `ec`,
`rtdetr`, `rfdetr`, `dfine`, `deim` and `deimv2`, two more than the seven
families verified end to end below. Each
`libreyolo/models/<family>/utils.py` re-exports from it, so existing import
paths keep working.

## Try the CPU-only wheel first

Most people asking for this want to avoid a multi-gigabyte install, and the
size is concentrated in one place: the default `torch` wheel bundles CUDA. A
CPU-only build is a fraction of that and needs no special install path.

<code-tabs name="install" />

The CPU-only option keeps every LibreYOLO feature: training, validation, every
task, every family, the CLI. Take the lightweight path when you want zero torch
on the machine, not merely less of it.

## What the lightweight install covers

| | |
|---|---|
| Task | Detection |
| Format | ONNX |
| Entry point | `OnnxBackend` |
| Interface | Python library |

Seven families were verified on this path: [YOLOv9](/docs/models/yolov9),
[YOLO-NAS](/docs/models/yolo-nas), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) and [DEIM](/docs/models/deim), counting each
family's variants with it.

That is the verified scope, not a boundary the library enforces. Other tasks
and other families are simply outside what was checked: some will pull torch
when you call them, and a few may happen to work. Treat anything beyond this
list as untested rather than as supported or as broken.

Inside it, results are identical to the normal install, not merely close. Each
family was exported to ONNX and run twice, once normally and once with torch
blocked; boxes, scores and classes matched exactly. A parity test in the suite
keeps that contract from drifting.

## The five things that catch people

**Use `OnnxBackend`, not the model classes.** `LibreYOLO9("model.onnx")` still
requires torch, because `LibreYOLO9` is itself an `nn.Module` subclass. This is
the likeliest mistake, since every other page in these docs loads a model
through its class or through `LibreYOLO()`.

**Export somewhere else.** Producing the `.onnx` file requires torch, so the
lightweight machine cannot make one. Export on a development or CI machine and
ship the artifact to the slim target.

**Results carry numpy arrays.** `result.boxes.xyxy` is an `ndarray` here. The
containers accept either type so the attribute names are unchanged, but code
that calls `.cpu()` or `.numpy()` on a result will fail.

**A single image returns a single `Results`.** `predict()` returns one
`Results` for one image and a list for several. Indexing a single result with
`[0]` selects the first detection, not the first image, which silently gives
you a one-box result instead of raising.

**The CLI will not work.** `typer` and `click` are not in the four packages, so
the `libreyolo` command is unavailable. This is a library install.

## Predict

<code-tabs name="predict" />

Swap `onnxruntime` for `onnxruntime-gpu` to run on CUDA. The four packages are
the ones a full torch-free `predict()` actually imports, recorded during the
call rather than reasoned about. `opencv-python-headless` stands in for the
declared `opencv-python`: same module, no GUI libraries, smaller on disk.

Of the remaining declared dependencies, `requests` is needed only to load an
image from a URL, `pycocotools` and `scipy` are validation and evaluation, and
`typer` and `click` are the CLI.

## This list will drift, by design

The package list above is correct for the release named at the top of this
page. `--no-deps` opts you out of dependency resolution, so nothing checks it
for you, and a later release may import something not listed here.

If you hit a `ModuleNotFoundError`, you already understand the technique:
install the missing package. That is the intended maintenance model rather than
a bug report. This path is best effort and is not a separately supported
distribution, which is also why there is no second lightweight package on PyPI
and no plan for one.

To confirm your environment is really torch-free rather than quietly falling
back to an installed copy, assert it:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

That check is worth keeping in CI for the slim image. Without it, an
environment that happens to have torch will pass every test and tell you
nothing.
