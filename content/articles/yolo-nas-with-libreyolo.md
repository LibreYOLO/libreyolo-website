---
title: How to run YOLO-NAS with LibreYOLO
description: YOLO-NAS is one of the most accurate real-time detectors available but the original repo is not maintained. Here is how to run it with LibreYOLO.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, yolo-nas, object-detection, tutorial]
faq:
  - q: "Is super-gradients still maintained?"
    a: "No. Development stopped after Deci was acquired by NVIDIA, so issues go unanswered and pinned dependencies like torchmetrics 0.8 conflict with newer stacks. LibreYOLO is a maintained alternative that loads the same YOLO-NAS weights."
  - q: "Can I use YOLO-NAS commercially?"
    a: "The YOLO-NAS weights carry Deci's non-commercial license, and it applies no matter which library loads them. Research and non-commercial inference are fine; for a commercial product you need to talk to Deci. LibreYOLO's own code is MIT; the restriction is upstream."
  - q: "When is super-gradients still the right choice?"
    a: "Two cases: CoreML export, which LibreYOLO does not support for YOLO-NAS, and TensorRT, where super-gradients has documented, tested paths while LibreYOLO's YOLO-NAS TensorRT support is untested."
---

YOLO-NAS posts some of the best accuracy-speed numbers of any real-time detector: the large variant hits 52.2 mAP on COCO while still running in real time. It has 3 size variants (S, M, L) as well as two task variants (detection, pose).

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters — visionanalysis.org">
</iframe>

Getting YOLO-NAS running from the official repo (supergradients) takes a bit more work than it should:

* Install super-gradients, which pulls in hydra, omegaconf, boto3, tensorboard, and a pinned `torchmetrics==0.8` that conflicts with newer stacks,

* Import from three or four different submodules to load a model and run a prediction,

* Unwrap the output through `ImagesDetectionPrediction` to get your boxes and scores.

Another problem of the official repo is that it is not maintained at all since the company that created supergradients (DeciAI) was acquired by nvidia. Which means that there are the typical problems such as lack of support and conflicts with newer stacks. \
\
LibreYOLO is a solid supergradients alternative, since it is maintained and it loads the same YOLO-NAS weights through the same easy to use API you use for every other model:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

The S, M, and L variants all work for detection and pose.

LibreYOLO can export YOLO-NAS to ONNX, TorchScript, OpenVINO, NCNN, and TFLite which represents a better support than the original library.

## Where super-gradients is still the right choice

**CoreML export**: LibreYOLO does not support CoreML for YOLO-NAS. The CoreML exporter is limited to a different set of families. If you are shipping on Apple devices and need a `.mlpackage`, stay with super-gradients.

**TensorRT**: super-gradients has documented, tested TensorRT paths for YOLO-NAS with known batch-size quirks spelled out. LibreYOLO's TensorRT support for YOLO-NAS is untested.

## A note on the weights

The weights are Deci's, not LibreYOLO's . The same non-commercial license applies regardless of which library you use to load them. LibreYOLO links to Deci's CDN and does not rehost anything. For research and non-commercial inference you are fine. For a commercial product you need to talk to Deci either way.

LibreYOLO's code is MIT. The weight restriction is upstream, not something LibreYOLO adds. We plan to train from scratch YOLO-NAS models in the future.

## Try it

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)   # xyxy coordinates
print(results[0].boxes.conf)   # confidence scores
```

LibreYOLO is MIT-licensed, runs on Linux, Mac, and Windows, and covers GPU, Apple Silicon, and plain CPU with no code change. One API spans RF-DETR, D-FINE, DEIM, YOLO-NAS, segmentation, pose, depth, and much more.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
