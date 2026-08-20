---
title: "Best YOLOv8 Alternatives in 2026"
description: "RF100-VL measures how well a detector fine-tunes off COCO. RF-DETR and YOLO-NAS score higher than YOLOv8. How to run both in LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "What is the best YOLOv8 alternative in 2026?"
    a: "RF-DETR-S for a GPU. It scored 60.41 mAP50-95 on LibreYOLO's RF100-VL campaign. The RF100-VL paper reports 56.9 for YOLOv8m. YOLO-NAS-S is the convolutional option, at 58.00."
  - q: "Does RF-DETR transfer better than YOLOv8?"
    a: "On RF100-VL, yes. LibreYOLO measured RF-DETR-S at 60.41 and RF-DETR-M at 61.13. Roboflow reports 60.2 and 61.2. The paper reports 56.9 for YOLOv8m."
  - q: "Can I use YOLO-NAS commercially?"
    a: "Deci's pretrained weights are non-commercial. The architecture is Apache-2.0. See the YOLO-NAS article for the license details."
---

COCO is the usual way to pick a detector. Another angle is how well the model fine-tunes on other data. [RF100-VL](https://arxiv.org/abs/2505.20612) measures that: a COCO-pretrained checkpoint is trained for 100 epochs on each of 100 datasets, then scored on each test split. The number is the mean mAP50-95.

YOLOv8m is 56.9 there, from the [paper appendix](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). Two models we ran score higher: RF-DETR and YOLO-NAS. Raw runs are in [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

RF-DETR-S is 60.41 and RF-DETR-M is 61.13 in [LibreYOLO's campaign](/articles/rf100vl-benchmark). [Roboflow reports](https://rfdetr.roboflow.com/latest/learn/benchmarks/) 60.2 and 61.2. Nano through Large are Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Docs](/docs/models/rf-detr) | [Weights](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Run `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S is 58.00, the convolutional option if you want a YOLO-style CNN. Deci's published weights are non-commercial. LibreYOLO hosts none of them. Details: [YOLO-NAS is still maintained](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Docs](/docs/models/yolo-nas) | [Run `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Install with `pip install "libreyolo[rfdetr]"` for RF-DETR, or `pip install libreyolo` for YOLO-NAS. Campaign page: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
