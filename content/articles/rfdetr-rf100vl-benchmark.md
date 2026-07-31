---
title: "Benchmarking RF-DETR on RF100-VL: one model, one hundred datasets"
description: We are running RF-DETR and 14 other detection families across all 100 RF100-VL datasets. Here is the benchmark explained, an interactive map of the datasets, and the first leaderboard.
date: 2026-07-31
author: Xuban
tags: [LibreYOLO, RF-DETR, RF100-VL, benchmark, object-detection, roboflow]
faq:
  - q: "What is RF100-VL?"
    a: "RF100-VL is a benchmark of 100 real-world object detection datasets collected from Roboflow Universe, spanning seven domains: aerial, document, flora and fauna, industrial, medical, sports, and a misc category. It was introduced by Roboflow to measure how well detectors generalize beyond COCO, and it is the benchmark used in the RF-DETR paper."
  - q: "What is the RF100-VL protocol in this report?"
    a: "For each model family we fine-tune the COCO-pretrained checkpoint for 100 epochs on each dataset's train split, then evaluate on that dataset's test split. We report mAP50 and mAP50-95 averaged over the datasets that completed, plus the median training time per dataset."
  - q: "Which model won the RF100-VL sweep?"
    a: "In the preliminary sweep, RF-DETR S leads with 0.562 mAP50-95 and 0.784 mAP50, ahead of EC, DEIMv2, RT-DETRv4 and DEIM, all near 0.55. YOLO-NAS S is the best accuracy-per-hour trade-off at 0.544 mAP50-95 with about 37 minutes of training per dataset."
  - q: "Why do RTMDet and PicoDet score near zero on RF100-VL?"
    a: "They are not broken: both train and produce detections on a subset of datasets, but the default fine-tuning recipe fails to converge on most of the 100 mostly small datasets. The heads use plain BCE which also conflicts with AMP autocasting. A validated small-dataset recipe for these two families is still open work."
---

COCO has 80 classes and a decade of overfitting behind it. If you want to know whether a detector actually works in the real world, you need a harder test. That test is **RF100-VL**: 100 datasets pulled from Roboflow Universe, crowdsourced from real projects, and grouped into seven domains: aerial, document, flora and fauna, industrial, medical, sports, and everything else.

This is the benchmark the RF-DETR paper used to claim state-of-the-art real-world accuracy, and it is the benchmark we are running across **15 detection families** in LibreYOLO: one representative model per family, fine-tuned for 100 epochs on each dataset's `train` split, then evaluated on its `test` split. 15 families x 100 datasets means 1,500 training runs. This article is the living report.

## The 100 datasets

Chest x-rays next to conveyor belts. Coral reefs next to circuit boards. Aerial sheep counting next to invoice parsing. The spread is the point: a model that wins here did not memorize a domain.

<rf100vl-hero></rf100vl-hero>

Every dot is one dataset, colored by domain, orbiting one ring per domain. Hover (or tap) a dot and the background shows you an annotated sample from that dataset. The rings are sized by domain: industrial and flora/fauna carry the most datasets, sports the fewest.

A few things that make RF100-VL genuinely hard:

* **Tiny training sets.** Many datasets have a few hundred images. Transfer learning quality matters more than architecture tricks.
* **Dense scenes.** Some datasets average over 100 objects per image (circuit elements reach ~255), which stresses query-based detectors and GPU memory alike.
* **Wild class imbalance and odd aspect ratios.** Drone footage, thermal cameras, microscopy, documents, screenshots.
* **No room for per-dataset tuning.** With 100 datasets you pick one recipe per family and live with it.

## The protocol

For each family we take the COCO-pretrained checkpoint at the smallest practical size (S for most, R18 for RT-DETR, T for YOLOv9 E2E) and fine-tune 100 epochs at 640 px (576 for RF-DETR). Then `model.val(split="test")`. We track three numbers per family: **mAP50**, **mAP50-95**, and the **median wall-clock training time** per dataset, because accuracy you cannot afford to train is not accuracy.

<rf100vl-results></rf100vl-results>

## Reading the leaderboard

The table stings a little, and that is the point: on 100 diverse datasets, architecture choice moves the needle by tenths of mAP, not percentage points.

* **The transformer detectors own the top.** RF-DETR S leads at 0.562 mAP50-95, with EC, DEIMv2, RT-DETRv4 and DEIM packed within 0.014 behind it. They are also 3 to 5 times slower to train than the YOLO-style families.
* **YOLO-NAS S is the efficiency king.** 0.544 mAP50-95 at roughly 37 minutes per dataset, finishing all 100 datasets. If you retrain weekly, this row matters more than the crown.
* **Two families collapse.** RTMDet and PicoDet floor near zero on most datasets. They are not broken models; their default recipes simply do not converge on small-data fine-tuning. Recipe, not architecture, is the bottleneck there.

The full per-dataset breakdown, failure analysis, and training curves land in the final report. Until then, the leaderboard above updates as the sweep progresses.
