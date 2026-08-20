---
title: "YOLO-NAS Is Still Maintained"
description: "Run, train, validate and export YOLO-NAS in LibreYOLO. SuperGradients last released in April 2024. We plan to train new weights from scratch to drop Deci's license."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "Is YOLO-NAS abandoned?"
    a: "SuperGradients last released 3.7.1 in April 2024. LibreYOLO supports YOLO-NAS prediction, training, validation and export on current PyTorch."
  - q: "Does Ultralytics maintain YOLO-NAS?"
    a: "Ultralytics supports inference, validation and export. It does not support training."
  - q: "Can pretrained YOLO-NAS weights be used commercially?"
    a: "Deci's pretrained weights retain their upstream non-commercial terms, whichever library loads them. Training from random initialization avoids those checkpoints. We also plan to publish from-scratch COCO weights."
---

YOLO-NAS is still a useful detector. Its original home, [SuperGradients](https://github.com/Deci-AI/super-gradients), last released 3.7.1 in April 2024. LibreYOLO maintains prediction, training, validation and export for it.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO supports S, M and L for detection, plus pose. Checkpoints download from Deci's public CDN under names like `LibreYOLONASs.pt`. LibreYOLO hosts none of them, and Deci's non-commercial terms still apply. SuperGradients pins PyTorch between 1.9 and 1.13; LibreYOLO requires 2.4 or newer. Detection export covers ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch and Core AI. CoreML is not supported.

To train without a Deci checkpoint:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

We also plan to train YOLO-NAS from scratch on COCO and publish those weights.

Install with `pip install libreyolo`. The [YOLO-NAS docs](https://www.libreyolo.com/docs/models/yolo-nas) have the model details. There is a longer [SuperGradients alternative](/articles/supergradients-alternative) article.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
