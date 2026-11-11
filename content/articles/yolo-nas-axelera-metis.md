---
title: "YOLO-NAS on Axelera Metis: INT8 keeps 99% of the accuracy"
description: LibreYOLO compiles YOLO-NAS detection, oriented boxes and pose for Axelera's Metis. On the same images the INT8 model on the card keeps about 99% of the FP32 accuracy.
date: 2026-11-11
author: Xuban
tags: [LibreYOLO, YOLO-NAS, Axelera, Metis, INT8, quantization, edge-ai]
---

![Axelera Metis M.2 Max card](/articles/yolo-nas-axelera-metis/axelera-metis-m2-max.jpg)

*Axelera Metis M.2 Max. Photo: Axelera AI*

The next LibreYOLO release (1.8.0) exports YOLO-NAS to Axelera's Metis, an INT8 accelerator on an M.2 card. Detection, oriented boxes and pose use the same call as any other export format:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
path = model.export(format="axelera")     # INT8 .axm for the Metis

metis = LibreYOLO(path)
results = metis("guggenheim-bilbao.jpg")  # runs on the card
metrics = metis.val(data="coco.yaml")
```

We ran the FP32 model and the INT8 model on the card on the same images. The card keeps about 99% of the accuracy on average and never less than 97.5%. Hardware: one Metis M.2 Max with Axelera's Voyager SDK 1.9.0.

<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:24px 0 8px">
<figure style="margin:0"><img src="/articles/yolo-nas-axelera-metis/detect-fp32.jpg" alt="YOLO-NAS-m FP32 detection on the Guggenheim Bilbao photo" style="margin:0"><figcaption style="font-size:13px;color:#7a7974;margin-top:6px;text-align:center">FP32 in PyTorch</figcaption></figure>
<figure style="margin:0"><img src="/articles/yolo-nas-axelera-metis/detect-metis-int8.jpg" alt="YOLO-NAS-m INT8 detection on the Metis on the same photo" style="margin:0"><figcaption style="font-size:13px;color:#7a7974;margin-top:6px;text-align:center">INT8 on the Metis</figcaption></figure>
</div>

*YOLO-NAS-m at confidence 0.35. Person 0.97 in both.*

## Results

![Share of FP32 mAP50-95 kept by the INT8 model on the Metis: between 97.5% and 100.5% for the eight YOLO-NAS models](/articles/yolo-nas-axelera-metis/accuracy-kept.svg)

| Model | Task | Images | FP32 | Metis INT8 | Kept |
| --- | --- | --- | --- | --- | --- |
| YOLO-NAS-s | detect | COCO val2017, 1000 | 0.4549 | 0.4519 | 99.3% |
| YOLO-NAS-m | detect | COCO val2017, 1000 | 0.4942 | 0.4893 | 99.0% |
| YOLO-NAS-l | detect | COCO val2017, 1000 | 0.5036 | 0.5025 | 99.8% |
| YOLO-NAS-R-s | OBB | DOTA-v2.0 val, 300 tiles | 0.3844 | 0.3747 | 97.5% |
| YOLO-NAS-R-m | OBB | DOTA-v2.0 val, 300 tiles | 0.4141 | 0.4105 | 99.1% |
| YOLO-NAS-R-l | OBB | DOTA-v2.0 val, 300 tiles | 0.4327 | 0.4348 | 100.5% |
| YOLO-NAS-pose-m | pose | COCO persons, 1000 | 0.7810 | 0.7722 | 98.9% |
| YOLO-NAS-pose-l | pose | COCO persons, 1000 | 0.7818 | 0.7727 | 98.8% |

The values are mAP50-95. These are subsets of the benchmarks, so the FP32 values differ a little from the published ones.

## What it took

### Output order

Our ONNX graph emits box regression first and class scores second. The compiled model returns them the other way round and nothing fails. You just decode the wrong tensors. The export now matches outputs by shape before compiling and the backend reads them by name.

### Calibration

| Calibration ranges for YOLO-NAS-s detect | Accuracy kept |
| --- | --- |
| Histogram (the SDK default) | 97.1% |
| Min-max (what LibreYOLO uses for detection) | 99.3% |

The histogram ranges clipped the positive class logits at stride 8, the level that finds small objects, so no score there could pass 0.5.

### Pose

![YOLO-NAS-pose-s keypoint mAP on the Metis after each fix: 0.076 by default, 0.659 with output windows, 0.706 with equalization and 0.734 with an activation cap, against 0.757 in FP32](/articles/yolo-nas-axelera-metis/pose-s-fixes.svg)

With the default settings pose-s scored 0.076 against 0.757 in FP32. Some raw pose outputs span thousands of units (one goes from -1939 to +22) and INT8 has 256 steps. One step on the keypoint offsets was about 21 pixels.

The fix is an output window. We rescale each output so the range that matters fills the INT8 range, clamp the rest with ReLU6, and undo the rescale on the host. Evening out channel ranges in the keypoint branch and capping one activation in the class branch at its 99.5th percentile gave the rest. 32-bit outputs, SmoothQuant, per-channel depthwise quantization and person-only calibration images did not help.

Pose is supported for m and l. The two smaller sizes lose more.

<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:24px 0 8px">
<figure style="margin:0"><img src="/articles/yolo-nas-axelera-metis/pose-fp32.jpg" alt="YOLO-NAS-pose-m FP32 keypoints on the Guggenheim Bilbao photo" style="margin:0"><figcaption style="font-size:13px;color:#7a7974;margin-top:6px;text-align:center">FP32 in PyTorch</figcaption></figure>
<figure style="margin:0"><img src="/articles/yolo-nas-axelera-metis/pose-metis-int8.jpg" alt="YOLO-NAS-pose-m INT8 keypoints on the Metis on the same photo" style="margin:0"><figcaption style="font-size:13px;color:#7a7974;margin-top:6px;text-align:center">INT8 on the Metis</figcaption></figure>
</div>

*YOLO-NAS-pose-m at confidence 0.35. Person 0.68 in FP32 and 0.72 on the card.*

## A val bug in every export format

While validating oriented boxes the FP32 reference scored about zero. `val()` on an exported YOLO-NAS-R model used the detection preprocessing: an ONNX export of YOLO-NAS-R-s scored 0.00003 mAP where the PyTorch model scores 0.46. This had nothing to do with Axelera and the fix covers every export format. The same check found bugs of the same kind in exported YOLOv3, YOLOv4 and YOLOv7 models. All three fixes ship in 1.8.0.

## Licence

LibreYOLO's code is MIT. The pretrained YOLO-NAS weights are Deci's: non-commercial use only and no redistribution. LibreYOLO downloads them from Deci's CDN and hosts no compiled `.axm` files, so you compile your own, and what you compile from those weights carries the same terms. The Voyager SDK is Axelera's software and you install it yourself.

## Other families

The same approach passes on 29 sizes across 10 families: YOLO-NAS, YOLO9, YOLOX, YOLOv3, YOLOv4, YOLOv7, PP-YOLOE, ResNet, MobileNetV4 and FCN. The full table ships in `docs/axelera.md`. Axelera export is a preview in 1.8.0. There are no speed numbers here because we haven't run a clean benchmark on all four cores yet.

![Axelera Metis M.2 Max card seen from above](/articles/yolo-nas-axelera-metis/axelera-metis-m2-max-top.jpg)

*The Metis M.2 Max from above. The Metis chip is the silver square. Photo: Axelera AI*
