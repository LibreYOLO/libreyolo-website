---
title: "RF100-VL: benchmarking detectors on one hundred real datasets"
description: We are running 15 detection families across all 100 RF100-VL datasets. Here is the benchmark explained, an interactive map of the datasets, and the first two verified results.
date: 2026-07-31
author: Xuban
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
faq:
  - q: "What is RF100-VL?"
    a: "RF100-VL is a benchmark of 100 real-world object detection datasets collected from Roboflow Universe, spanning seven domains: aerial, document, flora and fauna, industrial, medical, sports, and a misc category. It was introduced by Roboflow to measure how well detectors generalize beyond COCO. It is a dataset benchmark, not a model: any detector can be evaluated on it."
  - q: "What is the RF100-VL protocol in this report?"
    a: "For each model family we fine-tune the COCO-pretrained checkpoint for 100 epochs on each dataset's train split, then score that dataset's test split with pycocotools at maxDets 500. We report the unweighted mean of mAP50 and mAP50-95 across the 100 datasets, plus the median training time per dataset. A run only counts as a result when all 100 datasets completed."
  - q: "Which models have completed the RF100-VL sweep so far?"
    a: "Two: YOLOv9-S at 0.559 mAP50-95 and 0.814 mAP50, and YOLOv9-T at 0.540 mAP50-95 and 0.796 mAP50. Both trained and scored all 100 datasets with none skipped, at a median of roughly 33 minutes per dataset. The other families are still running and are not listed until they finish a full 100."
  - q: "Where can I check the RF100-VL numbers myself?"
    a: "Every artifact is published at huggingface.co/datasets/LibreYOLO/rf100-vl-results: the per-dataset training configs, per-epoch metrics, logs, GPU telemetry, scoring inputs and the submission JSON. Each run carries a manifest.json pinning the exact LibreYOLO and harness commits that produced it."
---

<under-construction></under-construction>

COCO has 80 classes and a decade of overfitting behind it. If you want to know whether a detector actually works in the real world, you need a harder test. That test is **RF100-VL**: 100 datasets pulled from Roboflow Universe, crowdsourced from real projects, and grouped into seven domains: aerial, document, flora and fauna, industrial, medical, sports, and everything else.

We are running it across **15 detection families** in LibreYOLO: one representative model per family, fine-tuned for 100 epochs on each dataset's `train` split, then evaluated on its `test` split. 15 families x 100 datasets means 1,500 training runs. This article is the living report.

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

For each family we take the COCO-pretrained checkpoint at the smallest practical size (S for most, R18 for RT-DETR, T for YOLOv9 E2E) and fine-tune 100 epochs at 640 px (576 for RF-DETR). Then `model.val(split="test")`, scored with pycocotools at maxDets 500. We track three numbers per family: **mAP50**, **mAP50-95**, and the **median wall-clock training time** per dataset, because accuracy you cannot afford to train is not accuracy.

One rule governs what appears below: a family is listed only after it has trained and scored all 100 datasets from a clean state under one set of commits. A partial sweep is a debugging artifact, not a result, so it is not shown here at any confidence level.

<rf100vl-results></rf100vl-results>

## Reading the results

Two families have finished a full 100 so far, and both are YOLOv9. That is not a ranking yet, so read it as two reference points rather than a podium.

* **Capacity buys less than you would guess.** YOLOv9-S carries 3.6 times the parameters of YOLOv9-T and converts that into 0.019 mAP50-95, from 0.540 to 0.559. On 100 diverse datasets, architecture and scale move the needle by hundredths, not percentage points.
* **Training cost is flat between them.** Both sit near 33 minutes per dataset in the median, so the larger model is close to free here. Fine-tuning on a few hundred images is dominated by fixed per-dataset overhead, not by model size.
* **The spread across datasets dwarfs the spread across models.** The gap between the two models is far smaller than the gap between the easiest and hardest datasets for either one. Which datasets resemble your problem matters more than which of these two you pick.

Every number above is traceable to a published run, and a family appears only once it has trained and scored all 100 datasets. The full per-dataset breakdown, failure analysis and training curves land in the final report; until then this page grows as each sweep finishes.
