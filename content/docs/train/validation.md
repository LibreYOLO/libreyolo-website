---
title: Validation and metrics
seo_title: "Validation and metrics in LibreYOLO"
description: "Run val() on any model, read the metric keys each task returns, choose an evaluation backend, and turn on a validation loss alongside the accuracy metric."
lead: "val() runs a model over a dataset split and returns a flat dictionary of metric keys and float values. The keys are literal strings, and which ones you get depends on the task, not the family."
keywords:
  - map50-95
  - coco evaluation
  - validation metrics
  - faster-coco-eval
  - pycocotools
  - validation loss
  - miou
  - panoptic quality
  - top1 accuracy
last_verified: "1.5.0"
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: On the test split
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml", split="test", batch=32)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Write COCO-format predictions
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
---

## Run a validation

`val()` takes the dataset and returns the metrics.

<code-tabs name="val" />

The return value is a plain `dict[str, float]`. Every key is literal, so read it
by name rather than by position.

The main arguments are `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` and `verbose`. `conf` defaults to
`0.001` and `iou` to `0.6`, both far looser than prediction defaults, because a
mAP sweep needs the low-confidence tail. `imgsz` defaults to the model's own
input size rather than a fixed number. `split` accepts `val`, `test` or `train`
and nothing else.

Any other field of the validation config passes through as a keyword argument,
including `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache`
and `save_plots`.

## Metric keys per task

Detection returns the COCO family of numbers:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Two of those are traps. `metrics/precision` and `metrics/recall` are aliases held
for backward compatibility: they carry the mAP 50-95 and AR@100 values, not a
precision and recall pair. Use the named keys.

Instance segmentation returns the mAP and AR figures above as mask numbers
under the unsuffixed keys, with the box versions under a `(B)` suffix and the
mask versions repeated under `(M)`. Precision and recall exist only in
suffixed form for this task, as `metrics/precision(B)`/`metrics/recall(B)` and
`metrics/precision(M)`/`metrics/recall(M)`, and both pairs carry the same
alias values as detect's: the `(B)` pair is box mAP50-95 and box AR@100, the
`(M)` pair is mask mAP50-95 and mask AR@100.

| Task | Keys |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, plus the size and recall breakdowns above |
| segment | mask versions of the detect keys above (unsuffixed keys are mask); `precision`/`recall` exist only as `(B)`/`(M)`, both aliased the same way |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L`, and the matching `keypoints_AR` keys |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, plus `(OBB)`-suffixed copies |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| point | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, plus a mAP sweep key |

`accuracy_top5` is really top-`min(5, num_classes)`, so on a three-class dataset
it is top-3, which every sample satisfies and which therefore reads 1.0.

The point task's sweep key is built from the distance thresholds, so with the
defaults it reads `metrics/mAP@[0.01:0.10]` and the single-threshold key reads
`metrics/mAP@0.01`. Passing `dist_thresholds` changes both strings.

Most tasks also return a `fitness` key, the single number best-checkpoint
selection uses. Detection, segmentation, pose and OBB do not; they are selected
on `metrics/mAP50-95`.

## Speed keys

Every validator adds timing:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

These are per-image milliseconds averaged over the run. They describe the machine
and settings you ran on, so a figure taken from them is only meaningful reported
with its hardware, batch size and precision.

## Evaluation backend

Detection and segmentation metrics are computed through a COCO evaluator, and
`faster_coco_eval=True`, the default, selects the C++ backend when the
`faster-coco-eval` package is installed. When it is not, the run falls back to
pycocotools with one warning per process:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Which backend actually ran is recorded on the model as `last_eval_backend`, and
the CLI reports it in its output for detection-style tasks. Set
`LIBREYOLO_FASTER_COCO_EVAL` to override the config value from the environment.

`iou_thresholds` is honored only on the OBB path. The COCO path evaluates through
its own fixed 0.50 to 0.95 sweep and ignores the value.

## Validation loss

By default validation reports accuracy only. `val_loss=True` also computes the
family's training objective on validation batches.

<code-tabs name="valloss" />

It emits `metrics/loss` plus one `metrics/loss/<component>` per term, weighted
exactly as training weights them, so the components sum to the total. Through a
logger they appear as `val/loss` and `val/loss/<component>`, and `libreyolo
monitor` overlays `metrics/loss` with `train/loss`.

The components are the family's own:

| Task | Families | Components |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

It is off by default because target assignment adds time and memory to
validation. The validator reuses the model output already produced for the
accuracy metric rather than running a second forward pass, it runs under
`no_grad` on the evaluation or EMA model, and under multi-GPU training it is
computed locally on rank 0 with no collectives. Best-checkpoint selection stays
on the accuracy metric.

Three things it deliberately does not do. It never includes contrastive-denoising
terms, because those need the ground truth at forward time and validation
forwards without it. It reports the evaluation-mode model, so where a family's
train and eval forwards genuinely differ, in BatchNorm statistics or stochastic
depth, the number reflects eval mode; that is the intended comparison. And a task
a family has not implemented it for raises a configuration error at setup rather
than quietly skipping:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO is the exception that changes nothing: its validator always computed this
loss, and `val_loss=True` only affects which keys it is published under.

Augmented validation and validation loss cannot be combined, and asking for both
raises.

## Files a validation writes

`val()` always writes `config.yaml` into its save directory, defaulting to
`runs/val/<model>_<size>_<timestamp>` when `save_dir` is not given.

<code-tabs name="json" />

`save_json=True` writes `predictions.json` for detection, and
`predictions_bbox.json` plus `predictions_masks.json` for segmentation. OBB does
not support it and says so.

`save_plots=True` writes into a `plots/` subdirectory. Detection gets
`box_metrics.png`, per-class AP and recall charts, precision-recall and
confidence curves, a confusion matrix, and annotated sample images when OpenCV is
installed. Segmentation adds the mask-side copies of each, and pose gets its own
metric and curve set. The other validators do not implement plots; classification,
semantic, panoptic, depth, normal, edge, restore, matte, OCR and point all write
nothing there. A plotting failure warns and never aborts the run.

## Validation during training

Training validates every `eval_interval` epochs against the dataset's `val`
split, and the metrics it produces are what drives `best.pt` selection, the
`patience` early stop, and the `val/` keys in every logger. The validation runs
on the EMA weights when EMA is on.

See [Hyperparameters](/docs/train/hyperparameters) for `eval_interval`,
`patience` and `save_plots`, and [Experiment loggers](/docs/train/loggers) for
where the numbers go.

## Related

- [Datasets](/docs/train/datasets) for the split keys and formats validators read.
