---
title: libreyolo train
seo_title: "libreyolo train command reference"
description: "Train a model from the command line: all 59 arguments with their defaults, how family defaults override them, and which arguments a family ignores."
lead: "Trains one model on one dataset and writes checkpoints, metrics and logs into a run directory. Every argument below has a default from the command definition, which a model family's own training config may replace."
keywords: [libreyolo train cli, libreyolo training command, yolo cli training, libreyolo train arguments, libreyolo dry run, libreyolo freeze layers]
last_verified: "1.5.0"
meta:
  - label: Command
    value: libreyolo train
    mono: true
  - label: Required
    value: data
    mono: true
  - label: Output
    value: "Checkpoints, metrics and logs under runs/train/exp"
snippets:
  examples:
    - label: Basic
      language: bash
      code: |
        # coco8.yaml ships with the package and downloads its 8 images on first use.
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640 batch=8
    - label: Check the resolved config first
      language: bash
      code: |
        # Prints what the run would use, including family defaults, and exits
        # without training or loading data.
        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10 dry_run=true
    - label: Named run with an explicit recipe
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
---

## Synopsis

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Arguments are `key=value` pairs, and POSIX form works too, so `epochs=50` and
`--epochs 50` are the same argument. Booleans accept `true` and `false`:
`amp=false` becomes `--no-amp` where the flag has a negative form.

## Arguments

### Model and data

| Argument | Default | Meaning |
|---|---|---|
| `data` | | Path to dataset YAML (YOLO format, e.g. `coco8.yaml`). Required |
| `model` | `yolox-s` | Model name or path to weights |
| `task` | | Explicit task override: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Use pretrained weights. `false` builds the architecture and trains from scratch |
| `allow_download_scripts` | `false` | Allow embedded Python in dataset YAML download blocks |

### Training loop

| Argument | Default | Meaning |
|---|---|---|
| `epochs` | `300` | Training epochs |
| `batch` | `16` | Batch size per device |
| `imgsz` | `640` | Training image size: `640` (square) or `480x640` (HxW) |
| `device` | `auto` | Device: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Dataloader workers |
| `cache` | `false` | Cache images to speed dataloading: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Random seed |
| `resume` | | Resume training: `true`, or a path to a checkpoint |
| `amp` | `true` | Automatic Mixed Precision |
| `amp_dtype` | `float16` | CUDA AMP dtype: `float16` or `bfloat16` |
| `cuda_graph` | `false` | Capture the training forward and backward into CUDA graphs. Single GPU, supported families only; the rest run eager |
| `lora` | `false` | LoRA fine-tuning, for the transformer families listed under Notes |
| `freeze` | | Freeze layers: an integer count, a list of indices, or module names |

### Distillation

| Argument | Default | Meaning |
|---|---|---|
| `distill_model` | | Teacher: a detector checkpoint, or a foundation-teacher id such as `dinov2` for backbone feature distillation |
| `dis` | | Distillation loss weight. The published default for the loss type when unset |
| `distill_loss_type` | `mgd` | Feature loss for detector teachers: `mgd`, `cwd`. Foundation teachers always use `feat_mse` |

### Optimizer

| Argument | Default | Meaning |
|---|---|---|
| `optimizer` | `sgd` | Optimizer: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Initial learning rate |
| `momentum` | `0.937` | SGD momentum, and the first-moment coefficient for the Adam optimizers |
| `weight_decay` | `0.0005` | L2 regularization |
| `nesterov` | `true` | Nesterov momentum |

### Scheduler

| Argument | Default | Meaning |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | LR schedule type |
| `warmup_epochs` | `5` | Warmup duration |
| `warmup_lr_start` | `0.0` | Initial warmup LR |
| `min_lr_ratio` | `0.05` | Minimum LR ratio |
| `lr_drop` | `100` | RF-DETR step LR drop epoch |

### Augmentation

| Argument | Default | Meaning |
|---|---|---|
| `mosaic` | `1.0` | Mosaic probability |
| `mixup` | `1.0` | Mixup probability |
| `hsv_prob` | `1.0` | HSV jitter probability |
| `flip_prob` | `0.5` | Horizontal flip probability |
| `degrees` | `10.0` | Rotation range, plus and minus, in degrees |
| `translate` | `0.1` | Translation ratio |
| `shear` | `2.0` | Shear angle |
| `mosaic_scale` | `(0.1,2.0)` | Mosaic scale range |
| `mixup_scale` | `(0.5,1.5)` | Mixup scale range |
| `no_aug_epochs` | `15` | Disable augmentation for the final N epochs |

### EMA

| Argument | Default | Meaning |
|---|---|---|
| `ema` | `true` | Exponential Moving Average |
| `ema_decay` | `0.9998` | EMA decay factor |

### Validation during training

| Argument | Default | Meaning |
|---|---|---|
| `val` | `true` | Validate during training |
| `eval_interval` | `10` | Validate every N epochs |
| `max_det` | `300` | Maximum predictions per image after validation NMS |
| `eval_max_det` | | COCO evaluator cap. The pycocotools AP@100 convention when unset |
| `faster_coco_eval` | `true` | Use the faster-coco-eval C++ backend for COCO metrics when installed; falls back to pycocotools |
| `save_plots` | `false` | Save final validation plots during training |
| `patience` | `50` | Early stopping patience. `0` disables it |

### Output

| Argument | Default | Meaning |
|---|---|---|
| `project` | `runs/train` | Output directory root |
| `name` | `exp` | Experiment name |
| `exist_ok` | `false` | Reuse existing output directory |
| `save_period` | `10` | Save checkpoint every N epochs |
| `log_interval` | `10` | Log loss every N batches |

### Agent flags

| Argument | Default | Meaning |
|---|---|---|
| `json` | `false` | JSON output to stdout |
| `quiet` | `false` | Suppress stderr |
| `dry_run` | `false` | Resolve and print the config without executing |
| `help_json` | `false` | Dump command schema as JSON and exit |

## Examples

<code-tabs name="examples" />

## Notes

### The defaults above are not always the values used

Every model family carries its own training config, and where that config
differs from the base one, its value replaces the command default for any
argument you did not set explicitly. Setting the argument yourself always
wins. `libreyolo cfg` prints the base defaults and the per-family overrides,
which is the way to see what a given family will actually use.

`imgsz` is the argument this matters most for. The command default is `640`,
which is not every checkpoint's native input: the published RF-DETR detection
sizes are 384, 512, 576 and 704, and the YOLOX `n` and `t` checkpoints are 416.
RF-DETR and DEIMv2 are handled by only forwarding `imgsz` when it was set
explicitly, so their own size stays in force otherwise. Other families are
handed the value as given and train at it. FOMO is the strict one: each size
accepts only its native input (96, 192 and 224), so a FOMO run needs `imgsz`
set to match or it stops with an error. RF-DETR also requires the value to
divide by its patch size times its window count, and reports the two nearest
legal sizes when it does not.

### Arguments a family ignores

Not every family reads every argument, and the augmentation ones are where that
shows. RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4 and DINOv2 train through
pass-through pipelines with no mosaic, no mixup and no affine warp, so
`mosaic`, `mixup`, `hsv_prob`, `degrees`, `translate`, `shear`, `mosaic_scale`
and `mixup_scale` reach nothing there. EC shares that pipeline but does read
`hsv_prob`, `degrees` and `translate` when its task is pose. The
classification families, SegFormer and NAFNet ignore that whole set and
`flip_prob` with it, because their flip runs at a fixed probability rather than
a configurable one. YOLO-NAS ignores `mosaic` alone, since it augments with an
always-on per-sample affine instead. RF-DETR ignores three more on top of that
list: `optimizer`, `momentum` and `nesterov`.

Setting one of these is not an error. The run logs a line to stderr naming the
family and the arguments it will ignore, then trains, and that line is the
authoritative list for the version installed. It is also the only signal, so a
scripted run with `quiet=true` suppresses the warning along with everything
else on stderr.

`val=false` is a related case. It sets `eval_interval` to `0` for most
families; RF-DETR cannot disable validation that way and logs that it ignored
the request.

### Other behavior worth knowing

`lora=true` is accepted by RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 and
v4, EC and ConvNeXt. Any other family exits with `config_unsupported` rather
than training without it.

`pretrained=false` combined with `resume` is refused for the families that
support scratch training, since the two ask for opposite things.

`mosaic` and `mixup` are the command-line spellings of the `mosaic_prob` and
`mixup_prob` config fields. On families whose mixup only applies to mosaic
samples, `mixup` above zero with `mosaic` at zero never fires, and the run says
so.

`dry_run=true` resolves the model reference, applies family defaults, and
prints the config it would train with. It does not load the dataset, so it is
the cheap way to confirm an argument reached the value you expected.

stdout carries the final result object; progress and warnings go to stderr.
The exit code is `0` on success, `2` for a usage or configuration error, `3`
when the dataset cannot be found or read, `4` when the model cannot be loaded,
and `1` for other runtime failures.

Related: [`libreyolo doctor`](/docs/cli/doctor) to check a dataset before
committing to a run, [`libreyolo monitor`](/docs/cli/monitor) to watch a run in
the browser, [`libreyolo val`](/docs/cli/val) to measure the result.
