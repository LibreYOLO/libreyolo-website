---
title: How to run RTMDet without mmdetection
description: RTMDet is one of the fastest accurate detectors around. The official install is broken on any PyTorch released in the last two years. Here is the alternative.
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
---

RTMDet is a real-time detector from OpenMMLab that achieves strong COCO accuracy while staying fast enough for deployment. The architecture is clean. The install is not.

Getting RTMDet running from the official source means assembling a four-layer OpenMMLab stack:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

The version pins are narrow. mmdet 3.3.0 demands `mmcv < 2.2.0`, but `mim install mmcv>=2.0.0` happily installs 2.2.0, which then fails a runtime assertion. You have to pin by hand.

Then there is the bigger problem. The newest mmcv wheel is 2.2.0, released April 2024 and never updated since. It only ships pre-built binaries up to **torch 2.4 / CUDA 12.1**. A fresh `pip install torch` today gives you PyTorch 2.12. That gap means mmcv has to compile its C++/CUDA ops from source: 10 to 30 minutes, a matching CUDA toolkit, `nvcc`, and a compatible C++ compiler. This is where the documented failures pile up:

* `ModuleNotFoundError: No module named 'mmcv._ext'` -- the compiled ops did not build or do not match,

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` on any RTX 30-series card,

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` -- the stack forces you back to Python 3.8,

* segmentation fault on import from a GCC version mismatch.

OpenMMLab's own FAQ tells you to install mmcv with pip instead of mim to avoid the version trap. Their Get Started page tells you to use mim. Both documents are official. They contradict each other.

Once the stack is running, inference still requires picking a config file whose name encodes the training recipe, plus a separately-downloaded checkpoint with a hash-stamped filename, wired through a registry you have to learn first.

LibreYOLO replaces all of it:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

No `mim`, no four-package version matrix, no source compilation, no config files, no checkpoint-hash hunting, no Python 3.8 pin. The weights are converted from the upstream OpenMMLab checkpoints and inference is bit-equivalent to mmdetection on the same checkpoint.

Five sizes are available: Tiny, Small, Medium, Large, and X. All run at 640 px.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

## Where the original is still the right choice

If you need to train or fine-tune RTMDet with the full MMDetection augmentation pipeline, or if you are already deep in the OpenMMLab ecosystem with mmcv working, stay there. LibreYOLO is the better path for inference and deployment.

## A note on the license

MMDetection and RTMDet are Apache 2.0. LibreYOLO's code is MIT. The weights carry the same Apache 2.0 terms from upstream. No commercial restrictions.

## Try it

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO is MIT-licensed, runs on Linux, Mac, and Windows, and works on GPU, Apple Silicon, and plain CPU with no code change. One API spans RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, segmentation, pose, depth, and more.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://libreyolo.com/docs)
