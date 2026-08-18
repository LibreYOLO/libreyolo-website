---
title: "RF100-VL: benchmarking detectors on one hundred real datasets"
description: We are running 17 detector configurations across all 100 RF100-VL datasets. Here is the benchmark explained, an interactive map of the datasets, and sixteen verified results.
date: 2026-08-18
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
faq:
  - q: "What is RF100-VL?"
    a: "RF100-VL is a benchmark of 100 real-world object detection datasets collected from Roboflow Universe, spanning seven domains: aerial, document, flora and fauna, industrial, medical, sports, and a misc category. It was introduced by Roboflow to measure how well detectors generalize beyond COCO. It is a dataset benchmark, not a model: any detector can be evaluated on it."
  - q: "What is the RF100-VL protocol in this report?"
    a: "For each model we fine-tune the COCO-pretrained checkpoint for 100 epochs on each dataset's train split, then score that dataset's test split with pycocotools at maxDets 500. We report the unweighted mean of mAP50 and mAP50-95 across the 100 datasets, plus the median training time per dataset. A run only counts as a result when all 100 datasets completed."
  - q: "Which models have completed the RF100-VL sweep so far?"
    a: "Sixteen models across five families have completed verified 100-dataset sweeps. RF-DETR-M leads at 0.6113 mAP50-95, ahead of RF-DETR-S at 0.6041. The completed field also includes RF-DETR-N, a LoRA fine-tune of RF-DETR-S, YOLO-NAS-S and M, EdgeCrafter-S, M and L, YOLOX-Nano, Tiny, S and M, and YOLOv9-T, S and M. Every listed model trained and scored all 100 datasets with none skipped. RF-DETR-L is the one configuration still running."
  - q: "Where can I check the RF100-VL numbers myself?"
    a: "Every artifact is published at huggingface.co/datasets/LibreYOLO/rf100-vl-results: the per-dataset training configs, per-epoch metrics, logs, GPU telemetry, scoring inputs and the submission JSON. Each run carries a manifest.json pinning the exact LibreYOLO and harness commits that produced it."
---

<under-construction></under-construction>

COCO has 80 classes and a decade of overfitting behind it. If you want to know whether a detector actually works in the real world, you need a harder test. That test is **RF100-VL**: 100 datasets pulled from Roboflow Universe, crowdsourced from real projects, and grouped into seven domains: aerial, document, flora and fauna, industrial, medical, sports, and everything else.

We are running **17 detector configurations** from LibreYOLO's supported families, each fine-tuned for 100 epochs on every dataset's `train` split, then evaluated on its `test` split. 17 models x 100 datasets means 1,700 training runs. This article is the living report.

## The 100 datasets

Chest x-rays next to conveyor belts. Coral reefs next to circuit boards. Aerial sheep counting next to invoice parsing. The spread is the point: a model that wins here did not memorize a domain.

Every planet orbiting at the top of this page is one of the 100 datasets: the thumbnail is a real annotated sample, the orbit is its domain. Hover a planet (or a domain in the legend) to explore. The rings are sized by domain: industrial and flora/fauna carry the most datasets, sports the fewest.

A few things that make RF100-VL genuinely hard:

* **Tiny training sets.** Many datasets have a few hundred images. Transfer learning quality matters more than architecture tricks.
* **Dense scenes.** Some datasets average over 100 objects per image (circuit elements reach ~255), which stresses query-based detectors and GPU memory alike.
* **Wild class imbalance and odd aspect ratios.** Drone footage, thermal cameras, microscopy, documents, screenshots.
* **No room for per-dataset tuning.** With 100 datasets you pick one recipe per family and live with it.

## The protocol

For each model we take a COCO-pretrained checkpoint and fine-tune it for 100 epochs, then run `model.val(split="test")`, scored with pycocotools at maxDets 500. We track three numbers per model: **mAP50**, **mAP50-95**, and the **median wall-clock training time** per dataset, because accuracy you cannot afford to train is not accuracy.

One rule governs what appears below: a model is listed only after it has trained and scored all 100 datasets from a clean state under one set of commits. A partial sweep is a debugging artifact, not a result, so it is not shown here at any confidence level.

<rf100vl-results></rf100vl-results>

## Reading the results

Sixteen models across five families have finished a full 100, and RF-DETR now holds the top two places. RF-DETR-M leads at 0.6113 mAP50-95, 0.0071 ahead of RF-DETR-S and 0.0312 ahead of the best non-RF-DETR campaign, YOLO-NAS-S. YOLO-NAS-S and M and EdgeCrafter-M stay a tight group at 0.5793 to 0.5800.

* **RF-DETR is the one ladder that scales cleanly.** N at 0.5736, S at 0.6041, M at 0.6113, monotonic all the way up. YOLOX also scales, from Nano at 0.4853 to M at 0.5701, but the rest do not: YOLO-NAS-S narrowly edges M, EdgeCrafter-M beats both S and L, and YOLOv9-S beats both T and M. Architecture size, pretraining and the family-specific recipe have to be judged together.
* **For RF-DETR, size mostly means resolution.** N, S and M carry 30.15M, 31.79M and 33.37M parameters, because the shared DINOv2 backbone dominates the count. What separates them is training resolution: 384, 512 and 576 pixels. The 0.0377 from N to M is bought mostly with pixels rather than weights, and it costs 24.6 minutes more per dataset.
* **The top two are also the slowest to train.** RF-DETR-M takes 69.7 minutes per dataset at the median, the most expensive campaign in the field, and RF-DETR-S 55.6. YOLOX-Nano is done in 18.7 and gives up 0.1260 mAP for it. Everything in the 0.55 to 0.58 band sits between 26.8 and 64.3 minutes.
* **LoRA trades more accuracy than it saves time.** RF-DETR-S fine-tuned with LoRA scores 0.5719, which is 0.0323 below the same checkpoint fully fine-tuned and drops it from second place to seventh, in exchange for 7.2 minutes per dataset at the median. On 100 small datasets the reason to reach for it is the memory ceiling, not the clock and not the score.
* **The spread across datasets still dwarfs the spread across models.** The 0.1260 gap from YOLOX-Nano to RF-DETR-M remains far smaller than the gap between the easiest and hardest datasets for any one model. Which datasets resemble your problem can matter more than which of these sixteen you pick.

Every number above is traceable to a published run, and a model appears only once it has trained and scored all 100 datasets. The interactive dataset view now includes all sixteen complete campaigns. One configuration is still training: RF-DETR-L at 704 pixels, the last of the 17. Failure analysis and training curves land in the final report; until then this page grows as the remaining sweeps finish.
