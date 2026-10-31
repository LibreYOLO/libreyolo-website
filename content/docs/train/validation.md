---
title: Validation and metrics
seo_title: "Validation and metrics in LibreYOLO"
description: "Run val() on any model, read the metric keys each task returns, choose an evaluation backend, and turn on a validation loss alongside the accuracy metric."
lead: "Validation runs a model over a dataset split through val() and returns a flat dictionary of metric keys and float values. The keys are literal strings, and which ones you get depends on the task, not the family."
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
  - confusion matrix
  - class-agnostic nms
last_verified: "1.6.0"
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
    - label: On another split
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

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
  confusion:
    - label: Confusion matrix
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        cm = metrics.confusion_matrix
        print(cm.matrix.shape)      # (81, 81): 80 classes plus background
        print(cm.class_accuracy())  # class name: share of found objects given that class
        tp, fp = cm.tp_fp()
  curves:
    - label: Per-class curves
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        box = metrics.box
        print(box.px.shape, box.f1_curve.shape)  # (1000,) and (classes, 1000)
        for index, f1 in zip(box.ap_class_index, box.f1):
            print(model.names[int(index)], round(float(f1), 3))
---

## Run a validation

`val()` takes the dataset and returns the metrics.

<code-tabs name="val" />

The return value is a flat `dict` of metric keys to numbers. Every key is
literal, so read it by name rather than by position. For detect, segment and
classify it is a `dict` subclass that also carries per-class results as
attributes, described below.

The main arguments are `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` and `verbose`. `conf` defaults to
`0.001` and `iou` to `0.6`, both far looser than prediction defaults, because a
mAP sweep needs the low-confidence tail. `imgsz` defaults to the model's own
input size rather than a fixed number. `split` accepts `val`, `test` or `train`
and nothing else. Without `data`, a trained checkpoint validates on the dataset
it was trained on; released weights carry none and need `data=`.

Any other field of the validation config passes through as a keyword argument,
including `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache`,
`save_plots`, `single_cls` and `agnostic_nms`. With `half=True` on CUDA,
validation runs under autocast in `amp_dtype`, and bfloat16 outputs are cast to
float32 before postprocessing; training with `amp_dtype="bfloat16"` validates
the same way.

`agnostic_nms=True` keeps only the highest-scoring box among boxes that overlap
above `iou`, whatever their classes. It runs on the family's finished
detections, after `classes=`, so mAP, the confusion matrix, the curves and the
per-image counts all score the same boxes. It covers detect and segment, and
any other task raises `ValueError`. The predict side is in
[Thresholds and filtering](/docs/predict/thresholds).

`single_cls=True` scores detection as one class. The ground truth is collapsed
to class 0, and the model's detections are kept to `classes=` in the model's
own label space, then relabelled 0, so a checkpoint trained on several classes
is scored as a one-class detector.

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

OBB's `metrics/precision` and `metrics/recall` are not aliases: they are the
real precision and recall at IoU 0.50, taken at the loosest operating point
(every prediction that survives `conf`, default `0.001`). The `(OBB)`-suffixed
copies repeat the same four values under a task-specific name, the same
convention as `(B)` and `(M)` above.

`accuracy_top5` is really top-`min(5, num_classes)`, so on a three-class dataset
it is top-3, which every sample satisfies and which therefore reads 1.0.
Classification results also expose the two values as the attributes `top1` and
`top5`, next to `confusion_matrix`.

The point task's sweep key is built from the distance thresholds, so with the
defaults it reads `metrics/mAP@[0.01:0.10]` and the single-threshold key reads
`metrics/mAP@0.01`. Passing `dist_thresholds` changes both strings.

Most tasks also return a `fitness` key, the single number best-checkpoint
selection uses by default. Detection, segmentation and OBB do not carry one;
their families are selected on `metrics/mAP50-95`, which their dicts do
return. Pose returns neither `fitness` nor `metrics/mAP50-95`; its trainers
set `best_metric_key` to `metrics/keypoints_mAP50-95` instead.

ImageFolder classification adds macro `metrics/precision`, `metrics/recall` and `metrics/f1`, averaged over classes present in validation targets. Top-1 remains default fitness. Detection also returns `metrics/best_conf` and `metrics/best_conf_f1`, choosing the micro-F1-optimal threshold at IoU 0.50, and puts class-name-keyed thresholds on `metrics.box.best_conf_per_class`. Equal-score detections stay grouped; ties choose the higher threshold. No positive F1 yields 0.0. Segmentation does not expose these threshold keys.

## Confusion matrix

Detect, segment and classify results carry `results.confusion_matrix`. It is
computed on every validation, whether or not plots are written, and its
`matrix` is an integer array indexed `matrix[predicted, true]`.

<code-tabs name="confusion" />

For detect and segment, predictions at confidence 0.25, or `conf` if higher,
are paired one to one with ground-truth boxes of any class at IoU 0.5, highest
IoU first. A pair counts at its predicted and true class, so a well-placed box
with the wrong class lands off the diagonal. The matrix has an extra
`background` row and column: an unpaired ground truth counts in the
`background` row, an unpaired prediction in the `background` column.
Segmentation pairs boxes, not masks. With `classes=`, ground truth of the other
classes is left out.

COCO crowd regions (`iscrowd=1`) follow the rule mAP applies. A prediction that
no ground truth of its class matched, lying at least half inside a same-class
crowd region (intersection over the prediction's own area), is ignored rather
than counted. `box.image_metrics` and the `visualize=True` images use the same
thresholds and the same crowd rule. YOLO-format labels have no crowd regions.

For classify, each image counts once, at its top-1 class against its label,
and there is no `background`.

| Method | Returns |
|---|---|
| `summary(normalize=False)` | One dict per predicted class, `{"Predicted": name, <true class>: value}`; with `normalize`, shares of each true class |
| `to_csv()`, `to_json()` | The summary as CSV or JSON text |
| `to_df()` | The summary as a Polars DataFrame; needs `pip install polars` |
| `tp_fp()` | Per-class true positives (the diagonal) and false positives (the rest of the predicted row) |
| `nonzero()` | The filled cells as aligned `(predicted, true, count)` arrays |
| `class_accuracy()` | Per class, the share of its instances given that class; for detection, of its found objects only |
| `plot(normalize=True, save_dir="")` | Writes `confusion_matrix_normalized.png`, or `confusion_matrix.png` with `normalize=False`, and returns the path |

Above 2,000 classes the dense classification matrix is not built: `matrix`,
`summary()`, the `to_*()` exports and `plot()` raise, while `tp_fp()`,
`nonzero()` and `class_accuracy()` still work.

The class is `libreyolo.validation.ConfusionMatrix`, built as
`ConfusionMatrix(nc, iou_thres=0.5, conf_thres=0.25, *, names=None, task="detect")`,
where `task` is `"detect"` or `"classify"`.

## Per-class precision, recall and F1

Detect and segment results carry each class's precision, recall and F1 across
confidence thresholds, matched at IoU 0.5.

<code-tabs name="curves" />

`box.px` holds the 1000 confidences, evenly spaced over `[0, 1]`.
`box.p_curve`, `box.r_curve` and `box.f1_curve` are `(classes, 1000)` arrays,
one row per entry of `box.ap_class_index`, the class indices that have ground
truth in the split, in ascending order. `box.p`, `box.r` and `box.f1` are the
per-class values at the confidence that maximizes the mean F1 over those
classes. Segmentation computes all of them from boxes. They are empty arrays
when the run produced no curves.

## Per-image box metrics

Detection and segmentation results stay dictionary-compatible and also expose `results.box.image_metrics`. Each filename maps to `precision`, `recall`, `f1`, `tp`, `fp` and `fn` using the visualization matching rule even when visualization is off. Segmentation counts boxes here. Duplicate basenames use full paths after the first occurrence. Zero denominators yield 0.0. These records are not gathered across distributed ranks.

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
| detect | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
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
`runs/val/<model>_<size>_<timestamp>` when `save_dir` is not given. `project`,
`name` and `exist_ok` choose it the way training does: `project/name`, with an
incremented suffix unless `exist_ok=True`.

<code-tabs name="json" />

`save_json=True` writes `predictions.json` for detection, and
`predictions_bbox.json` plus `predictions_masks.json` for segmentation. OBB does
not support it and says so.

`save_plots=True`, or its alias `plots=True`, writes into a `plots/`
subdirectory and needs matplotlib, which `pip install "libreyolo[plots]"`
installs. Detection gets `box_metrics.png`, per-class AP and recall charts,
precision-recall and confidence curves, the per-class F1 curves in
`f1_conf_box.png`, the confusion matrix as counts in `confusion_matrix.png` and
as shares of each true class in `confusion_matrix_normalized.png`, and annotated
sample images when OpenCV is installed. Segmentation adds mask-side copies of
the metric chart, the per-class charts and the precision-recall and confidence
curves; the F1 curves and the confusion matrix stay box-level. Pose gets its own
metric and curve set. Classification writes the two confusion matrix files,
except above 2,000 classes. The other validators do not implement plots;
semantic, panoptic, depth, normal, edge, restore, matte, OCR, OBB and point all
write nothing there. A plotting failure warns and never aborts the run.

`visualize=True` writes class-aware box TP/FP/FN images for detection and segmentation, or label-versus-top-1 images for ImageFolder classification, into `visualize/errors/` and `visualize/correct/`. Matching uses IoU 0.5 and confidence `max(0.25, conf)`. Defaults are `visualize=False`, `show_labels=True` and `show_conf=True`. Unsupported tasks and V-JEPA 2 clip validation reject visualization.

`plot_samples=8` limits the separate sample plot; 0 disables it and -1 retains all images. This does not change metrics or visualization output.

## Validation during training

Training validates every `eval_interval` epochs, and always after the final
epoch, against the dataset's `val` split, and the metrics it produces are what
drives `best.pt` selection, the `patience` early stop, and the `val/` keys in
every logger. The validation runs on the EMA weights when EMA is on. Its files
go to a `val/` directory inside the run. `val=False` turns validation during
training off, final epoch included.

See [Hyperparameters](/docs/train/hyperparameters) for `eval_interval`,
`patience` and `save_plots`, and [Experiment loggers](/docs/train/loggers) for
where the numbers go.

## Related

- [Datasets](/docs/train/datasets) for the split keys and formats validators read.
