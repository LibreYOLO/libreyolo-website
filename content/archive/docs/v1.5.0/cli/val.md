---
title: libreyolo val
seo_title: "libreyolo val command reference"
description: "Evaluate a checkpoint on a dataset split from the command line: every argument with its default, and the metric keys each task returns."
lead: "Evaluates one model against one dataset split and prints the metrics. The metric set depends on the model's task, and the numbers are the ones a benchmark row is built from."
keywords: [libreyolo val cli, libreyolo validation command, yolo cli evaluation, mAP50-95 command line, libreyolo val arguments]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo val
    mono: true
  - label: Required
    value: model, data
    mono: true
  - label: Output
    value: "Metrics on stdout. Plots and COCO JSON under runs/val/exp when asked for"
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Plots and COCO JSON
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Machine readable
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
---

## Synopsis

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Arguments are `key=value` pairs, and POSIX form works too, so `batch=8` and
`--batch 8` are the same argument.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `model` | | Model weights path or CLI name. Required |
| `data` | | Path to dataset YAML (YOLO format, e.g. `coco8.yaml`). Required |
| `data_dir` | | Direct dataset directory, bypassing the path in the YAML |
| `split` | `val` | Dataset split: `val`, `test`, `train` |
| `batch` | `16` | Batch size |
| `imgsz` | | Image size: `640` (square) or `480x640` (HxW). The model's own input size when unset |
| `conf` | `0.001` | Confidence threshold |
| `iou` | `0.6` | NMS IoU threshold |
| `max_det` | `300` | Max predictions per image after NMS |
| `eval_max_det` | | COCO evaluator cap. The pycocotools AP@100 convention when unset |
| `faster_coco_eval` | `true` | Use the faster-coco-eval C++ backend for COCO metrics when installed; falls back to pycocotools |
| `half` | `false` | FP16 inference |
| `amp_dtype` | `float16` | CUDA autocast dtype when `half=true`: `float16` or `bfloat16` |
| `save_json` | `false` | Save COCO-format JSON results |
| `save_plots` | `false` | Save validation plots: metrics, per-class AP, confusion matrix, samples |
| `workers` | `4` | Dataloader workers |
| `device` | `auto` | Device |
| `project` | `runs/val` | Output directory root |
| `name` | `exp` | Experiment name |
| `exist_ok` | `false` | Reuse output directory |
| `allow_download_scripts` | `false` | Allow embedded Python in dataset YAML download blocks |
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `verbose` | `true` | Verbose output |
| `help_json` | `false` | Dump command schema as JSON and exit |

## Examples

<code-tabs name="examples" />

## Notes

### What the metrics are

The printed set follows the model's task, and the JSON output uses the same
keys.

Detection, segmentation and oriented boxes report `mAP50`, `mAP50_95`,
`precision` and `recall`. Where a model predicts more than one output kind, the
per-kind groups appear alongside as `box_metrics`, `mask_metrics` and
`obb_metrics`, each carrying the same four keys.

Classification reports `accuracy_top1` and `accuracy_top5`. Point detection
reports `precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` and `mAP_sweep`.
Depth reports `abs_rel`, `rmse`, `delta1`, `delta2` and `delta3`. Semantic
segmentation reports `mIoU` and `pixel_accuracy`. Restoration reports `PSNR`
and `SSIM`.

The JSON result also carries `eval_backend`, naming the COCO evaluation library
and version that produced the numbers, so two runs can be compared knowing
whether the same backend scored both.

### Thresholds

The defaults here are evaluation defaults, not prediction defaults: `conf` is
`0.001` and `iou` is `0.6`, where [`libreyolo predict`](/docs/cli/predict) uses
`0.25` and `0.45`. Raising `conf` to a display threshold lowers recall and with
it the mAP, so a number produced that way is not comparable to a published one.

`imgsz` is unset by default, which means the model's own input size. Setting it
evaluates at the size given, which is how a checkpoint gets measured away from
its native resolution.

### Datasets that download

A dataset YAML whose `download` field is a URL fetches on first use with no
extra permission. One that carries an embedded Python download script needs
`allow_download_scripts=true`, and the command warns on stderr that local code
execution was enabled. The bundled `coco8.yaml` and `coco128.yaml` are URL
based, so they need nothing.

### Output and exit codes

stdout carries the metrics; progress goes to stderr. `json=true` prints one
object with `schema_version`, and `quiet=true` silences stderr.

The exit code is `0` on success, `2` for a usage or configuration error, `3`
when the dataset cannot be found, `4` when the model cannot be loaded, and `1`
for other runtime failures.

Related: [`libreyolo train`](/docs/cli/train), which runs this same evaluation
on its own schedule through `eval_interval`.
