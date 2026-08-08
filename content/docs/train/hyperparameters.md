---
title: Hyperparameters
seo_title: "Training hyperparameters in LibreYOLO"
description: "The train() arguments that matter: epochs, batch, lr0, optimizer, EMA, autobatch, gradient accumulation and resume, plus why defaults differ per family."
lead: "Every training argument is a field on a TrainConfig dataclass. The base class defines the field and its default; each model family subclasses it and overrides the defaults that its published recipe changes."
keywords:
  - train arguments
  - learning rate
  - batch size
  - autobatch
  - exponential moving average
  - gradient accumulation
  - resume training
  - early stopping patience
  - amp bfloat16
  - train config yaml
last_verified: "1.5.0"
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: Read a family's resolved defaults
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: |
        # Prints the train, val and predict defaults, including family overrides.
        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # batch=-1 probes GPU memory and resolves to a concrete power of two.
        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 4 micro-batches of 16 per optimizer step, effective batch 64.
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Load the interrupted run's checkpoint, then ask to resume.
        model = LibreYOLO("runs/train/exp/weights/last.pt")
        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Keys in the yaml are TrainConfig field names. Explicit kwargs win.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
---

## Setting arguments

`train()` takes keyword arguments and the CLI takes the same names in
`key=value` form.

<code-tabs name="train" />

Both paths end at the same place. The kwargs are handed to
`TrainConfig.from_kwargs()`, which builds the family's config dataclass.

## A typo does not raise

`from_kwargs()` drops any key that is not a field on the config and emits a
`UserWarning` naming it. Training then starts with the default in place:

```python
# UserWarning: Unknown training config keys (ignored): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Nothing fails, the run completes, and the learning rate was never what the caller
asked for. Read the warnings on the first epoch of a new recipe. The CLI is
stricter, because it validates flag names before the config is built, so a
misspelled CLI flag is rejected outright.

## Defaults are per family

`TrainConfig` defines the field and a base default. Each family subclasses it and
overrides what its published recipe changes, so there is no single correct answer
to "what is the default learning rate".

The base defaults are `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`,
`weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`,
`imgsz=640` and `amp=True`. Three examples of how far a family moves from that:

| Field | Base | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE and DEIM ship with `amp=False` because the D-FINE decoder clamps
activations at 65504, the largest finite float16 value. YOLO-NAS and FOMO also
default it off. The CLI's `--amp` flag defaults to `True` for every family, so it
counts as user-provided and overrides the family default; leave it alone unless
you mean to change it.

To read a family's real defaults rather than guessing:

<code-tabs name="defaults" />

## Batch size

`batch` is the global batch. Under multi-GPU training each rank loads
`batch // world_size`, so the number you pass is the number of images per
optimizer step regardless of how many GPUs are involved. See
[Multi-GPU training](/docs/train/multi-gpu).

`batch=-1` turns on autobatch. The trainer probes the model in training mode with
a real backward pass at powers of two, fits a line to the memory curve, and picks
the largest power of two strictly below the extrapolated value that fits within
60 percent of total VRAM.

<code-tabs name="autobatch" />

Probing in training mode with a backward pass is the point: an inference-mode
probe misses the retained activations and gradient tensors, which for a deep CNN
are several times the inference footprint. RF-DETR lowers the target fraction to
45 percent, because the probe's synthetic backward still underestimates what its
criterion and auxiliary decoder layers cost.

Autobatch is a CUDA feature. On CPU or MPS it logs one line and keeps the
default batch.

## Gradient accumulation

`nbs` sets the nominal, or effective, batch size. The trainer accumulates
`round(nbs / batch)` micro-batches per optimizer step.

<code-tabs name="accumulate" />

Left as `None`, the default, accumulation is off and training is unchanged.

## Learning rate and schedule

`lr0` is the initial learning rate and `optimizer` accepts `sgd`, `adam` and
`adamw`. `momentum` is SGD momentum or Adam's beta1, `weight_decay` is the L2
term, and `nesterov` applies to SGD.

The schedule is shaped by `scheduler`, `warmup_epochs`, `warmup_lr_start` and
`min_lr_ratio`. `no_aug_epochs` sets how many final epochs run without strong
augmentation, and several schedules use it to shape their tail as well, so it is
not purely an augmentation knob. What each family does with the augmentation half
of it is on [Augmentations](/docs/train/augmentations).

Some families add their own learning-rate knobs. `backbone_lr_mult` scales the
backbone group against the head, `clip_max_norm` sets gradient clipping, and
SegFormer uses `head_lr_mult` to run its decode head at ten times the backbone
rate. These live on the family's config subclass, not the base one.

## EMA

`ema=True` keeps an exponential moving average of the weights alongside the
trained ones. It is on by default everywhere except FOMO.

`ema_decay` is the target decay. The decay ramps in rather than starting at its
target: the effective value at update `n` is `ema_decay * (1 - exp(-n / tau))`
with `tau` defaulting to 2000, so early updates track the model more closely and
late updates smooth it. Family defaults range from `0.997` on YOLO-NAS pose
through `0.9998` on YOLOX to `0.9999` on YOLOv9 and the DETR line.

The EMA weights are what gets validated and what `best.pt` and `last.pt` carry.
The raw trained weights are also stored, under the `train_model` key, so a resume
continues from the trained trajectory rather than from the average.

## Precision

`amp=True` runs the forward pass under CUDA autocast. `amp_dtype` selects
`float16` (the default) or `bfloat16`; `fp16` and `bf16` are accepted spellings.

Float16 needs dynamic loss scaling and gets a live `GradScaler`. Bfloat16's wider
exponent range does not, so its scaler is constructed but disabled, which keeps
the optimizer path identical. Asking for bfloat16 on a CUDA device without
bfloat16 support raises at setup rather than degrading silently.

## Output, checkpoints and stopping

Runs are written to `project/name`, defaulting to `runs/train/exp`. With
`exist_ok=False`, the default, an existing directory gets an incremented suffix
instead of being overwritten.

`save_period` writes an extra `weights/epoch_<N>.pt` every N epochs, on top of
`weights/last.pt` after each epoch and `weights/best.pt` whenever the tracked
metric improves. `eval_interval` sets how often validation runs, and `patience`
stops the run after that many epochs without improvement, with `0` disabling
early stopping.

`cache` speeds up repeated epochs by holding decoded images in RAM (`True` or
`"ram"`) or as `.npy` files beside the sources (`"disk"`). Cached reads are
byte-identical to fresh ones. With dataloader workers, `"disk"` is the safer of
the two.

## Resume

`resume=True` continues an interrupted run. The checkpoint has to be loaded
first, because resume reads it from the model, not from a separate argument.

<code-tabs name="resume" />

Resume restores the trained weights, the optimizer state, the EMA weights and
update count, the best-metric tracking, the `GradScaler` scale, and the PyTorch,
CUDA and NumPy random states. It starts at the checkpoint's epoch plus one and
fast-forwards the schedule to that position.

Two things it will not do. `resume=True` cannot be combined with `pretrained`,
which raises. And when the checkpoint's best-metric key differs from the current
run's, best-metric tracking resets to zero with a warning rather than comparing
values that do not mean the same thing.

## Recipes in a file

`cfg=` loads a YAML mapping of `TrainConfig` field names and merges it under the
explicit keyword arguments, so a kwarg always wins over the file.

<code-tabs name="cfg" />

`size` and `num_classes` are stripped from the file, because the model instance
already owns them. There is no `--cfg` flag on the CLI; the file path is a Python
argument.

## Related

- [Datasets](/docs/train/datasets) for what `data=` accepts.
- [Augmentations](/docs/train/augmentations) for the augmentation knobs and which
  families honor them.
- [Layer freezing](/docs/train/layer-freezing) and [LoRA](/docs/train/lora) for
  training a subset of the weights.
- [Validation and metrics](/docs/train/validation) for what the run reports.
