---
title: "RF100-VL: how well do LibreYOLO models generalize?"
description: RF100-VL results measuring how YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter and RF-DETR generalize across 100 real-world datasets after fine-tuning.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL measures how well object detectors generalize across domains after fine-tuning. Its 100 datasets cover tasks such as detecting objects in aerial imagery, finding defects in industrial parts and analyzing medical scans. The benchmark tests how pretrained models adapt to the kinds of custom datasets used in computer vision projects.

A strong COCO score is useful when comparing pretrained checkpoints. Choosing a model for a new task also requires evidence of how well it can adapt to different objects and imaging conditions. Roboflow calls this [domain adaptability](https://blog.roboflow.com/rf-detr/), and evaluates it with RF100-VL alongside COCO accuracy and inference speed.

The results below compare YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter and RF-DETR. RF-DETR-L has the highest mean test AP in this comparison at 61.76. Per-dataset results, training recipes and checkpoints are available in the [public artifact archive](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main).

## The RF100-VL benchmark

In the fully supervised setting used here, a pretrained detector is fine-tuned separately on each dataset's training split. The selected checkpoint is then evaluated on that dataset's held-out test split. Generalization is measured after adaptation to each task; these are not zero-shot results.

The [RF100-VL paper](https://arxiv.org/abs/2505.20612) groups the datasets into seven domains: aerial, document, flora and fauna, industrial, medical, sports and other. The [released splits used for these results](https://huggingface.co/datasets/LibreYOLO/rf100-vl) contain 115,777 training images, 33,137 validation images and 14,237 test images. Training sets range from 72 to 8,791 images, with a median of 705. The paper's headline image count also includes images outside those released splits.

The score reflects the architecture, pretrained weights, training recipe and implementation together. It provides evidence of transfer across a range of tasks, while a representative evaluation on your own data remains necessary for model selection.

## Results

The comparison includes 17 configurations, each trained and evaluated on all 100 datasets. AP is shown on a 0 to 100 scale. The headline AP50:95 is the unweighted mean of the 100 dataset scores, so a small dataset counts as much as a large one.

<div>
<rf100vl-results></rf100vl-results>
</div>

These are single-seed results. We did not repeat the full sweep with multiple seeds, so small differences are not established improvements. The YOLOv9 rows describe the implementation used during this campaign, before the training fixes discussed below. LoRA is listed as its own configuration.

### RF-DETR transfers consistently across its four sizes

RF-DETR improves from 57.36 AP for N to 60.41 for S, 61.13 for M and 61.76 for L. L also has the highest mean in each of the seven domains. It wins 44 individual datasets; other configurations win the remaining 56. Its lead over M is 0.63 AP, and it scores higher than M on 67 datasets.

The results are close to [Roboflow's published RF-DETR table](https://rfdetr.roboflow.com/latest/learn/benchmarks/):

| RF-DETR configuration | This campaign | Roboflow reference |
|---|---:|---:|
| N | 57.36 | 57.7 |
| S | 60.41 | 60.2 |
| M | 61.13 | 61.2 |
| L | 61.76 | 62.2 |

That is a useful check on RF-DETR's behavior in LibreYOLO. It is not an exact reproduction: the starting checkpoints, resizing and some physical batch sizes differ, as documented in the methodology. Resolution also rises with size, from 384 for N to 704 for L. This sweep cannot tell us how much of the gain comes from resolution versus the other model differences.

### YOLO-NAS and YOLOX

YOLO-NAS-S scores 58.00 AP and YOLOX-M 57.01. Both exceed the 56.5 maximum in Roboflow's published YOLO11 table, although comparable inference latency remains unmeasured in this project.

YOLO-NAS-S and M differ by only 0.015 AP before rounding. Treat them as tied here. M wins more individual datasets, but a few larger losses bring its mean just below S. S also has a lower recorded median training time, 29.0 versus 60.9 minutes. Those timings came from the campaign, with the execution caveats below.

YOLOX improves from Nano through Tiny and S to M. Nano and Tiny ran at 416 pixels, while S and M ran at 640, so their differences also include resolution. The [YOLOX guide](/articles/yolox-with-libreyolo) and [YOLO-NAS guide](/articles/yolo-nas-with-libreyolo) cover using those families in LibreYOLO.

### LoRA saves median time here, with an accuracy cost

RF-DETR-S with backbone LoRA scores 57.19 AP, compared with 60.41 for full fine-tuning. Full fine-tuning wins on 95 datasets. The adapter recipe uses rank 16 with DoRA on the DINOv2 encoder; the projector, decoder and detection heads still train densely. The [published LoRA recipe](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/blob/main/rfdetr-s-lora/20260815-rfdetr-s-lora-1e542b7e/provenance/rfdetr-lora.json) records the details.

Median recorded training time falls from 55.6 to 48.4 minutes per dataset. Summed recorded job time does not fall: it is about 148 hours for LoRA versus 147 for full fine-tuning. Memory fallbacks, retries and the mix of dataset durations matter. This experiment supports an accuracy tradeoff for this adapter recipe; it does not establish a general compute saving from LoRA.

### YOLOv9 results and implementation limits

YOLOv9-M scored 52.71 AP, below T at 54.02 and S at 55.91. On `thermal-cheetah`, M was about 64 AP points below S. That pattern led to an [investigation of the training implementation](https://github.com/LibreYOLO/libreyolo/issues/795).

The campaign used single-head fine-tuning without the upstream training-only PGI auxiliary branch, top-left letterboxing, and a limit of 100 training labels per image. Dense datasets can exceed that limit. Subsequent changes restored PGI, recorded the letterbox convention in checkpoint metadata, increased the default label limit and added momentum warmup.

The archived scores remain measurements of the code that ran. They have not been replaced by results from the corrected trainer, and they should not be used to claim that YOLOv9-M inherently transfers worse than the smaller variants. EdgeCrafter also has a non-monotonic result, with M above S and L, but this sweep does not isolate its cause.

## Methodology

We followed the fully supervised RF100-VL setup. Each configuration starts from its public pretrained checkpoint and gets a separate fine-tune for each dataset. We kept a fixed evaluation protocol and pinned family recipes instead of tuning a model separately for each dataset.

| Setting | Campaign rule |
|---|---|
| Training budget | 100 epochs; early stopping disabled |
| Effective batch | 16 |
| Seed | 0; one completed run per configuration and dataset |
| Checkpoint selection | Validate every epoch; keep the best validation AP50:95 using EMA weights |
| Final evaluation | Evaluate the selected checkpoint on the test split |
| Scoring | Stock pycocotools on test, with `maxDets=500`; mean across all 100 datasets |
| Prediction thresholds | Confidence 0.001; NMS IoU 0.65 for NMS families |
| Recipe changes | Recorded and hashed; no per-dataset accuracy tuning |
| Data identity | Fixed dataset versions and released train/valid/test splits |

The evaluation cap matters for dense scenes. It also needs to be distinguished from a toolkit's default mAP calculation. [Appendix B of the paper](https://arxiv.org/html/2505.20612v4) explains the scoring differences. During this project we integrated faster-coco-eval for LibreYOLO's validation and compared it against stock pycocotools on saved predictions from all 100 test splits; the headline mean AP difference was zero.

### Augmentations and precision

The common protocol does not require an identical augmentation pipeline for every architecture. We retained the pinned recipe for each family, including its optimizer, learning-rate schedule and augmentation transitions.

| Family | Resolution | Training precision | Augmentation choices in the campaign recipe |
|---|---|---|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV and horizontal flip; mixup off; strong-augmentation tail of 15 epochs |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine transforms and horizontal flip; 15-epoch tail |
| YOLOX S/M | 640 | FP32 | Same family recipe, with the size-specific learning rate |
| YOLO-NAS S/M | 640 | FP32 | Mosaic off; mixup, HSV and horizontal flip; no final augmentation shutdown |
| EdgeCrafter S/M/L | 640 | FP16 | Mosaic and mixup off; horizontal flip; LibreYOLO family defaults |
| RF-DETR N/S/M/L and S LoRA | 384/512/576/704; LoRA at 512 | BF16 | Multi-scale training, expanded scales, crop/resize and horizontal flip |

The last 15 "no-aug" epochs in YOLO9 and YOLOX disable strong augmentations such as mosaic and mixup, rather than every image transform. YOLOX also enables its L1 loss branch at that transition. Keeping those transitions working mattered when we added graph capture.

Most recipes follow the family's reference choices, with deviations recorded alongside the run. EdgeCrafter had no public upstream fine-tuning recipe to reproduce, so its numbers measure the LibreYOLO defaults used in the campaign. The YOLOv9 implementation differences above also limit any claim of upstream recipe fidelity.

### Batch size and the RF-DETR reference

Roboflow's maintainer [documented physical batch 16 and accumulation 1](https://github.com/roboflow/rf-detr/issues/266#issuecomment-3110260396). Our RF-DETR N and S campaigns generally used that layout. M and L generally used physical batch 8 and accumulation 2 to fit the available memory. All retained effective batch 16.

The density policy and out-of-memory recoveries reduced the batch on some datasets. The archived records contain batch-4 fallbacks on 9 datasets for N, 16 for S, 7 for M, 8 for L and 19 for S LoRA. Three dataset runs also used DDP recovery. These choices are recorded in each run's statistics.

This distinction affects multi-scale training: a fresh scale is drawn per forward pass, so accumulation changes the number of scales contributing to an optimizer step. Equal effective batch does not make those recipes identical.

Our RF-DETR recipes also disclose public COCO starting weights versus Roboflow's private Objects365 checkpoints and OpenCV resizing without antialiasing. Roboflow's historical table predates a head-initialization fix in its own implementation. The [M recipe](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/blob/main/rfdetr-m/20260817-rfdetr-m-fc99ba9c/provenance/rfdetr-m-accum2.json) and [L recipe](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/blob/main/rfdetr-l/20260822-rfdetr-l-51d770a9/provenance/rfdetr-l-b8.json) preserve these deviations.

## Model sizes and training cost

The model selection emphasizes smaller variants to provide coverage within a fixed compute budget, with medium and large configurations and LoRA included as well. Each configuration requires 100 separate fine-tunes. The August 2026 runs used rented RTX 5060 Ti workers on Vast.ai; their 16 GB memory also constrained the physical batch sizes.

The workload was often limited by image loading, evaluation or CPU launch overhead. A faster GPU alone did not remove those costs. Running several datasets concurrently could improve throughput, but packing too many jobs onto a card or CPU could make the longest dataset finish later. The [campaign runbook](https://github.com/LibreYOLO/vision-analysis-benchmark/blob/rf100vl-harness/docs/rf100vl-operator-runbook.md) records the operating details.

The table's training times are medians of `wall_seconds` in the 100 training-stat records. They are useful for understanding this campaign, but the code and execution settings evolved between campaigns, jobs sometimes shared GPUs, and some runs resumed or retried. They are not controlled training-speed benchmarks or a reconstruction of the GPU bill.

We have not run these configurations through Roboflow's T4 single-artifact latency protocol. Training duration and inference latency answer different questions; neither should be substituted for the other.

## Inspect the datasets and reproduce the scores

The explorer contains all 17 configurations. Choose a model to inspect its per-dataset scores, or open a dataset to see the annotated sample and search for it on Roboflow Universe.

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

The [benchmark harness](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) handles training, checkpoint selection and test evaluation. Its public artifacts include the following records:

- `state/manifest.json`: installed code commits, recipe identity, dataset lock and hardware.
- `provenance/`: the recipe and dataset version lock. Some configurations share the same recipe file.
- `stats/`: per-dataset training settings, selected epoch, elapsed time and recovery details.
- `runs/`: training configuration, epoch metrics, logs and available telemetry.
- `eval/` and `submissions/`: per-dataset scores, prediction dumps and the final result.
- Checkpoints: the trained weights used for evaluation, with corrected YOLOX weights identified separately.

Use the record links in the results table to locate each configuration's submission. A run directory can contain stale copies of another model's submissions because earlier uploaders copied the whole submissions directory. The archive overview also predates the completed sweep. The result must come from that model's own authoritative run.

YOLOX Nano and Tiny require one extra note: their training histories live in the original folders, while corrected checkpoints and evaluation live in the replacement folders. A BatchNorm epsilon mismatch between training and reload had corrupted evaluation, especially for Nano. We repaired the checkpoints by folding the epsilon difference into the BatchNorm scale and rescored them. The archive keeps the [repair and supersession record](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/blob/main/yolox-nano/20260803-yolox-nano-c7bd2a8c/provenance/SUPERSEDES.md). Nano's earlier 36.01 AP is superseded by 48.53.

Each headline mean can be recomputed from the 100 dataset scores in its submission. The recipe hashes, dataset locks and per-dataset training statistics make the configuration and selection procedure inspectable. The results comprise 1,700 completed model-dataset fine-tunes, with no missing datasets in the published means.

## Thank you, Roboflow

Joseph Nelson offered GPU support after I wrote that I wanted to benchmark beyond COCO but could not afford the runs. Matvei Popov supplied reference setups, explained the evaluation settings and followed the results. Thank you both, and the Roboflow team, for the $500 budget and the time you put into helping this happen.

The sustained training workload gave us a reason to investigate problems that affected ordinary LibreYOLO users too. These are the main changes that came out of the campaign and the follow-up optimization work:

- **Evaluation and reproducibility.** We added configurable COCO evaluation caps, explicit AMP dtype selection and deterministic seeding before YOLO9's class-head adaptation. These made the benchmark protocol directly expressible in the library. [PR #664](https://github.com/LibreYOLO/libreyolo/pull/664)
- **Image loading.** Basic caching already existed. We extended it to cache the deterministic pre-augmentation resize, so eligible pipelines skip both decoding and resizing after the first pass, with identical cached pixels. Validation gained the same option. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677)
- **Validation overhead.** We reused validator instances, workers, buffers and parsed ground truth across epochs, then captured validation forwards where supported. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **CUDA graphs for training.** Capture support grew from YOLO9 and RF-DETR to 24 families. The work also fixed a capture race with DataLoader pin-memory threads that had killed real campaign jobs, preserved BatchNorm buffers during warmup and handled graph invalidation at training transitions. In a YOLO9-T experiment on 406 images (20 epochs, batch 8, 640 pixels), complete training fell from 428.4 to 367.7 seconds with the same reported AP. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **COCO scoring speed.** Dense-dataset evaluation was taking longer than training. The faster-coco-eval integration measured 15.6 times faster scoring overall on saved predictions from all 100 test splits. Of 1,400 metric values, 1,381 were bit-identical; the largest difference was 2.22e-16, and the headline AP difference was zero. This is evaluator speed, not a 15.6-fold training speedup. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **YOLOX correctness.** Rebuilding a model for a new class count had lost its BatchNorm settings. Fixing that removed the disagreement between training-time validation and reloaded inference, and the campaign supplied the checkpoints that exposed it. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700)
- **RF-DETR training speed.** We replaced slow L1-cost computation, reduced GPU-to-CPU synchronization, used fused AdamW, removed hidden device synchronization in shape checks and bounded matcher memory. A local RF-DETR-S experiment (512 pixels, batch 8, FP16) measured 266 to 234 ms per step including the training loop. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762)
- **Improvements shared with other families.** The RF-DETR investigation led to faster L1 matching in five more matchers, shared CUDA fused-optimizer construction, fewer YOLO9 loss/logging synchronizations and cached EdgeCrafter anchors and weighting tensors. [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765), [EC follow-up](https://github.com/LibreYOLO/libreyolo/commit/f96d5b96bcd37e72dce309faa85733970271af25)
- **Attention implementations.** We expanded the shared deformable-attention slot, integrated an existing Apache-2.0 Hub CUDA kernel and routed eligible transformer attention through PyTorch SDPA. Later work made the deformable-attention acceleration usable under FP16/BF16 autocast. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760)
- **An original Triton attention kernel.** We also wrote an in-tree multi-scale deformable-attention kernel for inference. It measured roughly four times faster on isolated RF-DETR-S attention shapes, and about a 7% end-to-end improvement for RF-DETR-N in the documented experiment. It does not implement the training backward path. [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **YOLOv9 training fidelity.** The anomalous M result contributed to the investigation and fixes for PGI, letterboxing, label limits and momentum warmup described above. The next campaign can measure the corrected training path. [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)
- **A reusable runner.** The harness gained parallel scheduling, multiple jobs per GPU, monitoring, telemetry, resume and OOM recovery, DDP lanes, artifact checks and cache cleanup. Other people can use that work to run the same benchmark. [Harness](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness)

The kernel and training-speed measurements above were separate experiments, largely on an RTX 5070 Ti. Their gains depend on the model, batch, precision and workload, and they should not be multiplied together. The linked changes record the checks and limits for each one.

Roboflow's support helped make these results public and enabled improvements to LibreYOLO's training and evaluation. Thank you for the generosity and guidance.
