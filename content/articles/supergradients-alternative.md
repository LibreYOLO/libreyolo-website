---
title: A SuperGradients Alternative for Real-Time Computer Vision
description: "SuperGradients stopped shipping after NVIDIA acquired Deci in 2024. LibreYOLO is a maintained alternative that runs the same YOLO-NAS weights: predict, train, and export under one MIT-licensed API."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "Is SuperGradients still maintained?"
    a: "No. The last release was 3.7.1 in April 2024, right before NVIDIA acquired Deci. Since then the repo has had no releases, issues go unanswered, and the documentation site went offline. It is not formally archived, but nobody is home."
  - q: "What is the best alternative to SuperGradients?"
    a: "For running YOLO-NAS, LibreYOLO loads the same Deci weights behind a maintained, MIT-licensed API and adds training, validation, and export. If you depend on SuperGradients' training recipes for other architectures, the old repo still runs once you pin its dependencies."
  - q: "Can I use YOLO-NAS commercially?"
    a: "Deci's pretrained weights are non-commercial no matter which library loads them. The architecture itself is permissively licensed, so training a YOLO-NAS from scratch on your own data gives you a model with no Deci checkpoint in its history."
  - q: "Does LibreYOLO replace everything SuperGradients did?"
    a: "Not everything. LibreYOLO has no CoreML export for YOLO-NAS and its TensorRT path for the family is untested, and SuperGradients' quantization-aware training recipes have no direct equivalent. For those, keep the old repo around."
---

SuperGradients was Deci's open-source training library and the home of YOLO-NAS, one of the most accurate real-time detectors ever released. In May 2024 NVIDIA acquired Deci, and SuperGradients went quiet. The last release, 3.7.1, landed on April 8, 2024. The documentation at supergradients.com went offline. Around 120 issues sit open, and the one titled ["Is this project dead?"](https://github.com/Deci-AI/super-gradients/issues/2062) has no answer from a maintainer, which is an answer of its own.

The repo is not archived, so at first glance it looks alive. In practice you feel the abandonment the moment you try to install it: `super-gradients` pins `torchmetrics==0.8`, which conflicts with current PyTorch stacks, and drags in hydra, omegaconf, boto3, and tensorboard along the way. Every month that passes, the pins fight harder with the rest of your environment, and no fix is coming.

None of this makes YOLO-NAS a worse model. The large variant still posts 52.2 mAP on COCO at real-time speed. The model is fine; its house burned down.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters - visionanalysis.org">
</iframe>

## Run the same weights in LibreYOLO

LibreYOLO loads the YOLO-NAS checkpoints directly from Deci's CDN, behind the same API it uses for every other model family. There are no hydra configs to write and no prediction wrapper to unwrap:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

The S, M, and L detection variants all work, and so does pose: swap in `LibreYOLONASs-pose.pt` and you get COCO keypoints back. Because every family returns the same `Results` object, comparing YOLO-NAS against RF-DETR or D-FINE on your own data is a one line change instead of a second codebase.

## Training and export, not just inference

SuperGradients was a training library first, so a real alternative has to train. LibreYOLO fine-tunes YOLO-NAS on your dataset with the same call it uses everywhere else:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

You can also train from a randomly initialized model, which matters for licensing (more on that below). Validation returns mAP metrics against any dataset in your format, and export covers ONNX, TorchScript, OpenVINO, NCNN, and TFLite. That is a wider export matrix than the original library ever shipped for YOLO-NAS. The full details are on the [YOLO-NAS docs page](https://www.libreyolo.com/docs/models/yolo-nas), and there is a shorter note that [YOLO-NAS is still maintained](/articles/yolo-nas-with-libreyolo).

## Where the old repo is still the right choice

Honesty section. Three cases where you should keep SuperGradients installed:

**CoreML.** LibreYOLO does not export YOLO-NAS to CoreML. If you ship on Apple devices and need a `.mlpackage`, SuperGradients still has the working path.

**TensorRT.** SuperGradients documented and tested its TensorRT flow for YOLO-NAS, batch-size quirks and all. LibreYOLO's TensorRT support for this family is untested.

**Quantization-aware training recipes.** YOLO-NAS was designed for INT8, and SuperGradients shipped the QAT recipes that made it shine there. LibreYOLO exports with INT8 and FP16 quantization but has no equivalent of those training recipes.

The repo still runs if you pin a 2024-era environment around it. You just should not build anything new on a foundation nobody maintains.

## A note on the weights

The pretrained YOLO-NAS weights are Deci's, released under a non-commercial license, and that license follows the weights into whichever library loads them. LibreYOLO hosts and mirrors none of them; the download comes from Deci's public CDN and prints Deci's terms before it starts. For research and non-commercial work you are fine either way.

If you need a commercial YOLO-NAS, there is now a clean route: the architecture itself is permissively licensed, so training from scratch on your own data produces a model that derives from no Deci checkpoint. LibreYOLO supports exactly that with `LibreYOLONAS(None, size="s")`.

## Try it

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO is MIT-licensed, runs on Linux, Mac, and Windows, and works on GPU, Apple Silicon, and plain CPU with no code change. One API spans YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet, and much more, across detection, segmentation, pose, classification, depth, and tracking.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://www.libreyolo.com/docs)
