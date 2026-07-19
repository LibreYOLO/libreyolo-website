---
title: Best Ultralytics Alternatives in 2026
description: "A practical guide to the best open-source alternatives to Ultralytics YOLO in 2026: their licenses, task coverage, and deployment limits, and where LibreYOLO, the most complete MIT-licensed YOLO library, fits."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Is Ultralytics YOLO free for commercial use?"
    a: "Only under AGPL-3.0, which requires releasing the source of the application you build on it, including networked services. Closed-source commercial use needs Ultralytics' paid commercial license. Permissive alternatives (MIT, Apache-2.0) avoid this."
  - q: "What is the most license-clean alternative to Ultralytics?"
    a: "For a permissive, ship-anywhere stack: LibreYOLO (MIT code), RF-DETR (Apache-2.0), and YOLOX (Apache-2.0). All avoid AGPL copyleft."
  - q: "What is replacing YOLO in 2026?"
    a: "Increasingly, real-time transformer detectors: RT-DETR, RF-DETR, and the D-FINE and DEIM line. They match or beat YOLO accuracy at similar latency on capable hardware. YOLO-style CNNs still win on small edge devices, where those transformers run poorly and lack a mature NCNN or CPU path."
  - q: "Can these run on a Raspberry Pi or an NPU?"
    a: "It depends on export. CNN detectors like YOLOX and RTMDet export to NCNN and run on a Pi, and some (YOLOX) reach the Hailo NPU. Transformer detectors like RF-DETR do not; they need a GPU or a strong CPU."
---

Disclosure: LibreYOLO is ours, and it is first on this list. Every other tool here is one we use in real work, and we note where it beats LibreYOLO.

Most teams leave Ultralytics for one of three reasons:

- **License.** Ultralytics' YOLO models are AGPL-3.0 (YOLOv5 since 2023, and v8 onward). The pattern is familiar: you build a product, ship it, and then a legal review or a customer flags the model, and the choice becomes open-source your whole application or buy a commercial license. AGPL's copyleft reaches your code the moment you distribute or serve it. This is a common reason people look elsewhere.
- **Openness.** One vendor owns the models, the training code, and the terms. Some teams want weights and code under permissive licenses they can build on without asking.
- **The model.** YOLO is not always the best architecture for the task. Real-time transformer detectors like RT-DETR, RF-DETR, and D-FINE can be more accurate at the same latency. The catch is that they live in scattered research repos, and, as you will see, LibreYOLO runs all three under one API, so this is a reason to switch libraries, not to leave for one.

Every tool below is judged on three things: a license you can ship, whether it installs and runs on a current PyTorch, and whether it exports to the hardware you actually deploy on.

One pattern repeats as you read: each alternative is good but individually painful. YOLOX barely installs, MMDetection is wheel-pin purgatory, the RT-DETR family is research code with no support line, Detectron2 has been frozen since 2021. LibreYOLO is first on this list because it already wraps most of them, maintained, MIT, on a current stack, behind one familiar API. It is less an entry on this list than the layer above it.

## The short version

| Tool | License | Tasks covered | Best for | Edge export |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (code) | Detection, segmentation, pose, classification, depth, gaze, tracking | The MIT hub for over 20 model families under one API | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Detection, segmentation, pose (preview) | Production detection and segmentation | ONNX, TFLite, TensorRT; no NCNN or Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Pretraining, distillation | Unlabeled data, distilling DINOv2/v3 | n/a (not a detector) |
| **YOLOX** | Apache-2.0 | Detection | Anchor-free real-time detection | install-hostile upstream, runs via LibreYOLO |
| **MMDetection / OneDL fork** | Apache-2.0 | Everything (research) | Breadth of architectures, reproducing papers | via export |
| **Detectron2** | Apache-2.0 | Detection, segmentation, keypoints | Mask R-CNN and the R-CNN family | manual |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Detection | State-of-the-art real-time DETRs | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Detection, segmentation, pose | Compact edge ViTs distilled from DINOv3 | research-grade |
| **RT-DETR (v1-v4)** | Apache-2.0 | Detection | Frontier real-time detection | ONNX, TensorRT |

## 1. LibreYOLO

License: MIT (code). Best for the most complete MIT-licensed YOLO library, and a drop-in Ultralytics alternative.

**LibreYOLO is the MIT-licensed YOLO library that runs every model under one API.** It is our project, and after the rest of this list its role should be clear: it is the hub that runs the alternatives. Nearly every painful repo above, YOLOX, RTMDet, RF-DETR, D-FINE, DEIM, the RT-DETR line, LibreYOLO already wraps behind one familiar, maintained API, so you get the model without the install archaeology or the license.

There is a reason that goes past convenience. YOLO started as open research: Joseph Redmon's Darknet, v1 through v3, was free for anyone to use and build on. The AGPL era enclosed it. LibreYOLO exists to make that work accessible again, the way its creators intended, permissively licensed, community-maintained, and not phoning home. Two stories, then.

**One: it is the MIT-licensed alternative to Ultralytics YOLO.** It keeps the API you already know, so your YOLO-format datasets and scripts port over with small changes, but it is MIT instead of AGPL: no copyleft reaching into your code, no commercial license to buy, and no telemetry phoning home the way Ultralytics does by default. If licensing is why you are here, that is the whole point. YOLO9 and RF-DETR are the heavily tested, production-ready defaults; the wider zoo is newer and clearly marked experimental, we would rather say so than pretend all of it is battle-hardened.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Two: it covers far more than Ultralytics does.** A single API spans over 20 model families, CNN YOLOs, transformer DETRs, ViT backbones, SAM, and more, not one vendor's lineup. And it is not only detectors:

- **Tasks:** instance and semantic segmentation, pose, classification, and oriented boxes, plus two Ultralytics does not have at all, monocular depth (Depth Anything V2) and gaze estimation (L2CS).
- **The practical layer:** multi-object tracking with persistent IDs (ByteTrack and OC-SORT), video inference with frame subsampling and live preview, tiled inference for drone and satellite imagery, a drag-and-drop browser UI (`libreyolo ui`), and a `doctor` command that checks your dataset before you burn a training run.
- **Training goes deep:** gradient accumulation, layer freezing, resume, test-time augmentation, LoRA/DoRA fine-tuning, multi-GPU, and TensorBoard/MLflow/Weights & Biases logging.
- **Export:** seven formats (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite) with INT8/FP16 quantization, embedded NMS, and embedded model metadata.
- **Runs anywhere:** accepts paths, URLs, PIL, NumPy, tensors, or raw bytes, and runs on CUDA, Apple Silicon (MPS), or plain CPU with no code change.

When a strong detector's own repo is abandoned or hard to install, LibreYOLO runs its weights, maintained, on a current stack.

## 2. RF-DETR

License: Apache-2.0 (Nano, Small, Medium, Large). Best for production detection and segmentation.

RF-DETR, from Roboflow and published at ICLR 2026, is the most production-ready model on this list. It is well engineered, it comes from a team that ships, and when you need detection or segmentation that works and that you can trust in a real system, it is a strong first choice. The `rfdetr` package and the Nano through Large weights are Apache-2.0. The larger XL and 2XL weights use Roboflow's PML license.

The limits are scope and deployment. It covers detection and segmentation, with a keypoint/pose model in preview, and no classification. And it does not reach constrained hardware. Official export is ONNX and TFLite; TensorRT is an ONNX conversion that needs a CUDA GPU. There is no NCNN path, which is the usual route onto a Raspberry Pi, and no Hailo NPU support. On a Pi 5 CPU it runs at roughly a second or more per frame, not real time. Roboflow's own advice for real time is to use a GPU, apply TensorRT, pick the smallest model, and lower the resolution. That is fine with a GPU or a Jetson-class board, and a problem for a cheap edge SoC or an NPU. Its published latencies are TensorRT FP16 on an NVIDIA T4, not CPU numbers.

## 3. Lightly

Best for unlabeled data, self-supervised pretraining, and distilling a DINO backbone.

Lightly is not a detector. It is how you get value from data you have not labeled, and it comes in two parts with different licenses.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** A framework of self-supervised learning components: losses, heads, augmentations, memory banks, and reference implementations of more than twenty methods (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE, and others). You write the training loop; it gives you everything to pretrain a representation from scratch on unlabeled images.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (with commercial and free-community options).** The turnkey one, and the workflow most people mean. Point it at a foundation model such as DINOv2 or DINOv3, point it at your unlabeled data, and it distills that backbone into a smaller model you can deploy (YOLO, RT-DETR, ViT, or a custom net) in a few lines, with no labels. Note the license: AGPL-3.0 is the same copyleft that sends people looking for Ultralytics alternatives, so read the terms if you plan to ship closed source, or take the commercial license.

Reach for Lightly when the bottleneck is a good backbone but no labels. It composes with the detectors on this list rather than competing with them. The newest of those show it too: RT-DETRv4 bakes vision-foundation-model distillation into training, and DEIMv2 builds DINOv3 backbones directly into the detector.

## 4. YOLOX

License: Apache-2.0. Best for anchor-free real-time detection, once it installs.

The upstream repo has been frozen since its last release, v0.3.0 in April 2022, with more than 700 open issues, and it is hostile to install on a 2026 stack. There is no `pyproject.toml`, and `requirements.txt` pins `onnx-simplifier==0.4.10`, which has no prebuilt wheels past Python 3.10, so on a modern interpreter it source-builds and needs cmake and a C++ toolchain. NumPy is left unpinned too, which invites NumPy 2.0 ABI clashes with older pycocotools and torch. Core detection runs once you resolve the dependency chain, but getting there is the tax.

The model itself is still good: an anchor-free real-time detector from Megvii, Apache-2.0 on both code and weights, and a reasonable baseline even if newer detectors have since passed it. Reviving an abandoned but sound repo is the kind of task a coding agent handles in an afternoon now, so the install state is less of a wall than it looks. LibreYOLO also runs the YOLOX weights on a current stack directly. Either way, the architecture is worth using. We wrote a fuller walkthrough [here](/articles/yolox-with-libreyolo).

## 5. The MMDetection universe

License: mostly Apache-2.0. Best for breadth of architectures and reproducing papers.

For years, MMDetection held the official implementation of nearly every detection paper: Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN, hundreds of configs. In 2026, OpenMMLab has wound the mm-series down. [MMDetection](https://github.com/open-mmlab/mmdetection)'s last release was v3.3.0 in January 2024, issues go unanswered, and the stack is stuck on the `mmcv` version pin, whose prebuilt wheels lag current PyTorch and CUDA, so a fresh install tends to break until you downgrade everything. We covered that pain [here](/articles/rtmdet-without-mmdetection).

A Dutch company, VBTI, keeps it alive. Their [OneDL fork](https://github.com/VBTI-development/onedl-mmdetection) re-publishes the whole stack under an `onedl-` prefix (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine`, and the rest) rebuilt for current PyTorch 2.x, CUDA, and Python 3.10+, which resolves the version pin. It is Apache-2.0 and actively maintained, with v3.5.1 in May 2026. For MMDetection's breadth without the install work, use this fork. It is a small team, so contribute back if you rely on it.

Two related tools:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) is in maintenance mode, with no tagged release since v0.6 in 2021, but it is reliable and still the cleanest source of Mask R-CNN and the R-CNN family for instance and panoptic segmentation.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, actively maintained) ships Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN, and Keypoint R-CNN. No modern YOLO or DETR, and you write your own training loop, but it is the zero-dependency baseline.

One caution for commercial use: **MMYOLO**, OpenMMLab's YOLO reimplementation hub, is GPL-3.0 rather than Apache-2.0, and has been frozen since August 2023.

## 6. The RT-DETR family

License: Apache-2.0 throughout. Best for state-of-the-art real-time detection with research-grade code.

The current frontier of real-time detection is a cluster of related transformer detectors rather than a YOLO. Most descend from RT-DETR, share much of the same backbone and encoder code, and are almost all Apache-2.0, so swapping between them is easy. Most are detection-only, with one exception (EdgeCrafter) that extends the same distillation ideas to segmentation and pose. The shared cost is that these are research repos: config-file training, ergonomics unlike Ultralytics, and no support channel. The models are strong, and ONNX and TensorRT export is usually first-class.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). Reformulates box regression as distribution refinement with self-distillation. Strong accuracy per latency (D-FINE-X reaches about 55.8 AP at roughly 13 ms on a T4), sizes N through X, and it is integrated into Hugging Face Transformers, which makes it the easiest here to load and fine-tune outside its own repo.
- **[DEIM and DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) is a training recipe (Dense O2O matching) that bolts onto D-FINE or RT-DETRv2 and roughly halves training time. DEIMv2 adds DINOv3 features and scales down to a sub-1M-parameter Atto tier for mobile; DEIMv2-S is the first sub-10M model past 50 AP. It is actively maintained, updated into 2026.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). The same lab's newest work, and the one entry here that goes past detection: it covers detection (ECDet), instance segmentation (ECSeg), and pose (ECPose), each in S/M/L/X sizes, all Apache-2.0. It adapts a large DINOv3-pretrained ViT as a task-specialized teacher and distills it into compact student backbones built for edge. The numbers are strong for the size: ECDet-S reaches 51.7 AP under 10M parameters on COCO alone, ECPose-X hits 74.8 AP (ahead of YOLO26Pose-X at 71.6), and its instance segmentation is competitive with RF-DETR's. It is brand new, so treat it as a research release, but it is a rare compact model that spans all three tasks.
- **[RT-DETR and RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). The original "DETRs beat YOLOs on real-time detection" work (CVPR 2024) and the root of the family: NMS-free, end-to-end, export-friendly. v2 is a 2024 bag-of-freebies update, a drop-in upgrade at the same latency. The Paddle original and the canonical PyTorch port share this repo, and it is in HF Transformers.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) and **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Peking University and Tsinghua, late 2025). v3 is PaddlePaddle-only. v4 is the current best of the line (X near 57 AP), is PyTorch, and works by distilling a vision foundation model into a lightweight detector at no added inference cost. Its codebase also reproduces D-FINE, DEIM, and RT-DETRv2 through config changes.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). A plain-ViT DETR presented as a transformer replacement for YOLO, sizes tiny through xlarge, with ONNX and TensorRT export. It also underpins the open-vocabulary OVLW-DETR. Lower recent activity; treat it as a stable research release.

LibreYOLO already wraps many of these (D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet, EdgeCrafter) behind one API, with YOLO9 and RF-DETR as its most tested models. For reference implementations and the newest research checkpoints, the source repos above are canonical.

## FAQ

**Is Ultralytics YOLO free for commercial use?**
Only under AGPL-3.0, which requires releasing the source of the application you build on it, including networked services. Closed-source commercial use needs Ultralytics' paid commercial license. Permissive alternatives (MIT, Apache-2.0) avoid this.

**What is the most license-clean alternative to Ultralytics?**
For a permissive, ship-anywhere stack: LibreYOLO (MIT code), RF-DETR (Apache-2.0), and YOLOX (Apache-2.0). All avoid AGPL copyleft.

**What is replacing YOLO in 2026?**
Increasingly, real-time transformer detectors: RT-DETR, RF-DETR, and the D-FINE and DEIM line. They match or beat YOLO accuracy at similar latency on capable hardware. YOLO-style CNNs still win on small edge devices, where those transformers run poorly and lack a mature NCNN or CPU path.

**Can these run on a Raspberry Pi or an NPU?**
It depends on export. CNN detectors like YOLOX and RTMDet export to NCNN and run on a Pi, and some (YOLOX) reach the Hailo NPU. Transformer detectors like RF-DETR do not; they need a GPU or a strong CPU.

## Try it

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO is MIT-licensed, runs on Linux, Mac, and Windows, and works on GPU, Apple Silicon, and plain CPU with no code change. One API spans RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter, and more, across detection, segmentation, pose, classification, depth, gaze, and tracking.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
