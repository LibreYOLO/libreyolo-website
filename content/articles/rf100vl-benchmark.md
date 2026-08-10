---
title: "RF100-VL: benchmarking detectors on one hundred real datasets"
description: We are running 15 detector configurations across all 100 RF100-VL datasets. Here is the benchmark explained, an interactive map of the datasets, and seven verified results.
date: 2026-08-09
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
faq:
  - q: "What is RF100-VL?"
    a: "RF100-VL is a benchmark of 100 real-world object detection datasets collected from Roboflow Universe, spanning seven domains: aerial, document, flora and fauna, industrial, medical, sports, and a misc category. It was introduced by Roboflow to measure how well detectors generalize beyond COCO. It is a dataset benchmark, not a model: any detector can be evaluated on it."
  - q: "What is the RF100-VL protocol in this report?"
    a: "For each model we fine-tune the COCO-pretrained checkpoint for 100 epochs on each dataset's train split, then score that dataset's test split with pycocotools at maxDets 500. We report the unweighted mean of mAP50 and mAP50-95 across the 100 datasets, plus the median training time per dataset. A run only counts as a result when all 100 datasets completed."
  - q: "Which models have completed the RF100-VL sweep so far?"
    a: "Seven models have completed verified 100-dataset sweeps. YOLO-NAS-S leads at 0.5800 mAP50-95, followed by YOLOX-M at 0.5701, YOLOv9-S at 0.5591, YOLOX-S at 0.5525, YOLOv9-T at 0.5402, YOLOX-Tiny at 0.5218, and YOLOX-Nano at 0.4853. Every listed model trained and scored all 100 datasets with none skipped."
  - q: "Where can I check the RF100-VL numbers myself?"
    a: "Every artifact is published at huggingface.co/datasets/LibreYOLO/rf100-vl-results: the per-dataset training configs, per-epoch metrics, logs, GPU telemetry, scoring inputs and the submission JSON. Each run carries a manifest.json pinning the exact LibreYOLO and harness commits that produced it."
---

<under-construction></under-construction>

COCO has 80 classes and a decade of overfitting behind it. If you want to know whether a detector actually works in the real world, you need a harder test. That test is **RF100-VL**: 100 datasets pulled from Roboflow Universe, crowdsourced from real projects, and grouped into seven domains: aerial, document, flora and fauna, industrial, medical, sports, and everything else.

We are running **15 detector configurations** from LibreYOLO's supported families, each fine-tuned for 100 epochs on every dataset's `train` split, then evaluated on its `test` split. 15 models x 100 datasets means 1,500 training runs. This article is the living report.

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

Seven models across three families have finished a full 100. YOLO-NAS-S is the current leader at 0.5800 mAP50-95, with YOLOX-M close behind at 0.5701.

* **Capacity helps, but family matters just as much.** Within YOLOX, scaling from Nano at 0.90M parameters to M at 25.28M moves mAP50-95 from 0.4853 to 0.5701. Within YOLOv9, 2.02M to 7.20M parameters buys only 0.0189, from 0.5402 to 0.5591. YOLO-NAS-S reaches 0.5800 with 19.02M parameters, beating the larger 25.28M YOLOX-M by 0.0099. The seven results turn the old one-pair anecdote into a consistent warning against choosing by parameter count alone.
* **Training cost now separates the sizes.** The median training time per dataset runs from 18.8 minutes for YOLOX-Nano to 48.0 minutes for YOLOX-M. YOLO-NAS-S reaches the best accuracy at a 29.0-minute median, so the largest compute bill is not buying the top result.
* **The spread across datasets dwarfs the spread across models.** The 0.0947 gap from YOLOX-Nano to YOLO-NAS-S is still far smaller than the gap between the easiest and hardest datasets for any one model. Which datasets resemble your problem can matter more than which of these seven you pick.

Every number above is traceable to a published run, and a model appears only once it has trained and scored all 100 datasets. The full per-dataset breakdown, failure analysis and training curves land in the final report; until then this page grows as each sweep finishes.
