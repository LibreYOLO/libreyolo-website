---
title: How to run YOLOX with LibreYOLO
description: YOLOX is Apache 2.0, commercially clean, and still competitive. The original repo is abandoned. Here is how to run it with LibreYOLO instead.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, yolox, object-detection, tutorial]
---

YOLOX is an anchor-free detector from Megvii with seven model sizes ranging from 0.91M parameters (Nano) to 99.1M (X). The large variant hits 49.7 mAP on COCO val. It is Apache 2.0 on both code and weights, which makes it one of the few competitive detectors you can ship in a commercial product without any restrictions.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolox-s%2Cyolox-m%2Cyolox-l%2Cyolox-x"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLOX accuracy vs parameters — visionanalysis.org">
</iframe>

The problem is the original library. `pip install yolox` gets you version 0.3.0 from April 2022 and has not moved since. Running inference is not a one-liner:

* You instantiate an `Exp` object to configure the model,

* load the checkpoint manually into the architecture,

* preprocess your image through `ValTransform`,

* pass it through the model and call `postprocess()` to get boxes and scores out of the raw output tensor.

The ONNX export uses `torch.onnx._export`, a private PyTorch API that was removed in PyTorch 2.x. The hard-pinned `onnx-simplifier==0.4.10` in `requirements.txt` conflicts with any newer `onnx` install. CoreML export does not exist at all. The repo has been in maintenance-only mode since mid-2022.

LibreYOLO gives you the same YOLOX weights through the same API you already know:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

All six sizes work: Nano (`n`), Tiny (`t`), Small (`s`), Medium (`m`), Large (`l`), Extra-large (`x`). Nano and Tiny run at 416 px input; the rest at 640 px. LibreYOLO handles that automatically.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

Export is broader than what the original library offers. LibreYOLO can export YOLOX to ONNX, TorchScript, CoreML, OpenVINO, NCNN, TFLite, and TensorRT. The original repo has no CoreML path and its ONNX export is broken on modern PyTorch.

Training works too:

```python
model.train(data="your_dataset.yaml", epochs=100, batch=16)
```

## A note on the license

YOLOX weights are Apache 2.0. That means no non-commercial restrictions, no need to contact anyone, no platform lock-in.

LibreYOLO's own code is MIT.

## Try it

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO is MIT-licensed, runs on Linux, Mac, and Windows, and works on GPU, Apple Silicon, and plain CPU with no code change. One API spans RF-DETR, D-FINE, DEIM, YOLO-NAS, YOLOX, segmentation, pose, depth, and more.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://libreyolo.com/docs)
