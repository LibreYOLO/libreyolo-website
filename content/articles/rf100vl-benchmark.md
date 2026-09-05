---
title: "RF100-VL: how well do LibreYOLO models generalize?"
description: RF100-VL results measuring how YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter and RF-DETR generalize across 100 real-world datasets after fine-tuning.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL asks how well a detector adapts beyond COCO. Each model is fine-tuned separately on 100 very different datasets, including infrared, X-rays, microscopy, sports, radio spectrum images, tiny objects and video games. We benchmarked 17 LibreYOLO configurations: **1700 fine-tunes in total.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Results

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L led this comparison at **61.76**. M followed at 61.13, then S at 60.41. The four RF-DETR results are close to [Roboflow's published numbers](https://rfdetr.roboflow.com/latest/learn/benchmarks/), which is a useful validation of RF-DETR training in LibreYOLO.

Size did not predict every result. YOLO-NAS-S and M are effectively tied at 58.00 and 57.99. EdgeCrafter-M scores 57.93, ahead of S at 55.99 and L at 56.11. We plan to investigate these regressions. This sweep is not a controlled scaling experiment: resolution, pretrained weights, precision and recipes vary.

RF-DETR-S with backbone LoRA scores 57.19, compared with 60.41 for full fine-tuning. Full fine-tuning wins on 95 datasets. Its recorded median is only seven minutes slower, while summed job time is almost unchanged.

The YOLOv9 rows predate important training fixes. The anomalous M result helped uncover missing PGI, a different letterbox convention and a 100-label training limit. The archived numbers describe the code that ran. They are not a verdict on the YOLOv9 architecture.

Choose a model below to inspect its scores across all 100 datasets.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Methodology

For every dataset, we trained on `train`, selected the best validation AP50:95 checkpoint and evaluated it once on `test`. The headline score is the unweighted mean of those 100 test results.

| Setting | Campaign rule |
|---|---|
| Training | 100 epochs; no early stopping |
| Effective batch | 16 |
| Seed | 0; one completed run per dataset |
| Checkpoint | Best validation AP50:95 using EMA weights |
| Test scoring | pycocotools with `maxDets=500` |
| Tuning | One pinned recipe per family; no per-dataset tuning |

Each family kept its own training recipe:

| Family | Resolution | Precision | Augmentation outline |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV, flip; strong augmentations off for the final 15 epochs |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine, flip; 15-epoch tail |
| YOLOX S/M | 640 | FP32 | Same family recipe, with a size-specific learning rate |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV and flip; mosaic off |
| EdgeCrafter S/M/L | 640 | FP16 | Flip; LibreYOLO family defaults |
| RF-DETR N/S/M/L and S LoRA | 384/512/576/704; LoRA at 512 | BF16 | Multi-scale, crop/resize and flip |

These are single-seed results. Small differences are not established improvements. RF-DETR M and L used gradient accumulation to fit the available memory, and some dense datasets needed smaller physical batches. The public RF-DETR recipes also start from COCO checkpoints, while Roboflow used private Objects365 checkpoints for its historical table.

Training times are campaign records, not controlled speed benchmarks. Jobs sometimes shared GPUs, retried or resumed. We did not run the optional T4 single-artifact latency protocol. YOLO-NAS uses Deci's separately licensed [non-commercial pretrained weights](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md).

## What the workload improved

Small tests can prove that training starts. They do not reproduce weeks of continuous fine-tuning across many architectures and datasets. At this scale, slow paths, memory pressure and correctness problems became hard to miss.

- **Image loading and validation.** We cached the deterministic resize before augmentation, reused validator workers and buffers, and graphed validation forwards where supported. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **Training CUDA graphs.** Capture support grew from YOLOv9 and RF-DETR to 24 families. We also fixed a DataLoader pin-memory race and graph invalidation at training transitions. A YOLOv9-T test fell from 428.4 to 367.7 seconds with the same reported AP. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **COCO scoring.** Dense-dataset evaluation sometimes took longer than training. The faster-coco-eval integration scored saved predictions from all 100 test splits in 8.4 seconds instead of 131.4. Of 1,400 metric values, 1,381 were bit-identical; the largest difference was 2.22e-16 and headline AP was unchanged. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR and shared training paths.** Faster L1 matching, fewer device synchronizations, fused AdamW and bounded matcher memory reduced an RF-DETR-S step from 266 to 234 ms in the recorded test. The same investigation improved five other matchers, YOLOv9 logging and EdgeCrafter tensor reuse. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Attention.** We routed eligible attention through PyTorch SDPA, integrated an Apache-2.0 CUDA implementation and fixed mixed-precision execution. We also wrote an in-tree Triton deformable-attention kernel for inference. It measured about four times faster in isolation and about 7% faster end to end for RF-DETR-N in the documented tests. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Training correctness.** We fixed YOLOX BatchNorm reconstruction and restored YOLOv9 PGI, letterboxing metadata, label capacity and momentum warmup. The benchmark supplied the checkpoints and failure patterns that exposed both problems. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Those numbers come from separate tests on different workloads. They should not be multiplied into one headline speedup. LibreYOLO is now faster and more reliable for real training workloads than it was before this campaign.

## Harness and artifacts

The public [training harness](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) schedules datasets across GPUs, resumes interrupted jobs, handles OOM recovery and records the recipes, dataset versions, logs, checkpoints and predictions.

The [run-rf100vl-benchmark skill](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) gives AI coding agents the protocol and Vast.ai operating instructions. The [result archive](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) contains every published run.

## Thank you, Roboflow

Joseph Nelson offered GPU support after I wrote that I wanted to benchmark beyond COCO but could not afford the runs. Matvei Popov shared the reference setups, explained the evaluation settings and followed the results as they arrived.

That support let us validate LibreYOLO training across 17 configurations, publish evidence people can use when choosing a model, release the harness and stress the library until its weak points became obvious.

Thank you, Joseph, Matvei and the Roboflow team, for the compute, time and guidance.
