---
title: Upgrading to 1.5.0
seo_title: "Upgrade LibreYOLO 1.4.0 to 1.5.0"
description: "The four code changes 1.5.0 requires, the three changes that move metrics, and the smaller behavior shifts worth knowing before you compare runs."
lead: "Nothing was removed from the public model API: every class and function that worked in 1.4.0 still imports. Four arguments changed shape, and three defaults move numbers you may be comparing against."
keywords: [libreyolo upgrade, libreyolo 1.5.0 migration, allow_experimental removed, libreyolo breaking changes, yolox bn eps, faster-coco-eval default]
last_verified: "1.5.0"
meta:
  - label: Applies to
    value: 1.4.0 to 1.5.0
  - label: Code changes required
    value: Four, all narrow
  - label: Results that move
    value: COCO backend, YOLOX BN eps, D-FINE multi-scale
  - label: Public API removals
    value: None
---

This page is about upgrading LibreYOLO itself. If you are looking for how to
load a checkpoint from an upstream project, that is
[import existing weights](/docs/migrate), a different subject.

The full entry for the release is the [changelog](/docs/changelog). What
follows is only the part that asks something of you.

## Code changes you must make

### `allow_experimental=True` no longer exists

The acknowledgement gate is gone, along with the
`ddp_aware(experimental_key=...)` mechanism behind it. EC, RTMDet, PicoDet and
FOMO training and export previously required the argument, so any script that
trains one of those families is affected.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: delete the argument
model.train(data="data.yaml", epochs=100)
```

There is no deprecation shim. A call that still passes it raises `TypeError`.
`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` was removed with it. The
`get_download_notice()` hook survives, and is still overridden by MiDaS,
SegFormer and YOLO9-P2.

Support levels are still published, they are just no longer an argument: see
[stability tiers](/docs/reference/stability-tiers).

### The export tier `"experimental"` no longer exists

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Code branching on the tier string should read `"available"` where it read
`"experimental"`. `BaseExporter` no longer emits a `RuntimeWarning` for those
formats. The per-format state is listed in the
[export matrix](/docs/reference/export-matrix).

### `pretrained=False` with `resume` is now rejected

The combination previously proceeded incoherently. It now raises:

```
ValueError: pretrained=False cannot be combined with resume.
```

Pick one. `pretrained=False` starts from a fresh seeded initialization, which
in 1.5.0 works for every trainable family rather than three of them, and
`resume` continues an interrupted run from its checkpoint. Both are documented
under [training](/docs/train).

### CLI `--imgsz` is a string, not an int

Narrower than it sounds. Both of these are unaffected:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # still fine
```

```python
model.predict("img.jpg", imgsz=640)   # still fine
```

Only code that calls the [CLI](/docs/cli) command functions directly from
Python needs to change, because `predict`, `train` and `val` widened `--imgsz`
from `int` to `str` so it can accept rectangular sizes:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, and "480x640" now works too
```

`train`'s default is now the string `"640"`. `export --imgsz` was already a
string, and `profile` is unchanged.

## Numbers that change

Three changes move metrics at default settings. If you track results across
versions, read these before you compare a 1.5.0 run against a 1.4.0 one.

### faster-coco-eval is the default COCO metrics backend

`val()` and per-epoch training validation now compute COCO metrics with the
faster-coco-eval C++ backend instead of pycocotools.

The switch was decided on measured parity across all 100 RF100-VL test splits:
1381 of 1400 metric values bit-identical, maximum deviation 2.22e-16, headline
deltas exactly 0, at 15.6x faster overall and 56x on detection-dense datasets.
Your numbers should not move. They are produced by a different implementation
all the same, which is the reason this is on the list.

pycocotools stays the automatic fallback when faster-coco-eval is not
installed. To force it:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` does the same thing globally. The backend
actually used is logged at INFO, exposed as `model.last_eval_backend` after
`val()`, and included as `eval_backend` in the
[CLI](/docs/cli/val) JSON payload. Install the fast path with
`pip install libreyolo[fast-eval]`.

### YOLOX checkpoints trained before 1.5.0 need an eps override

This is the trap in the release. Read it if you have fine-tuned
[YOLOX](/docs/models/yolox).

YOLOX specifies BatchNorm `eps=1e-3` and `momentum=0.03`. Until 1.5.0 those
values were applied as a post-hoc fixup that did not survive the class-count
rebuild `train()` performs when your dataset's `nc` differs from the
checkpoint's. Such a fine-tune trained and reported in-training validation at
torch's default `eps=1e-5`, then reloaded for inference at `1e-3`: the same
tensors under different normalization.

Regular-conv sizes barely move. Depthwise `n` moves a lot, because its
per-channel `running_var` is small enough for eps to dominate. On RF100-VL
`ball`, the same nano checkpoint scores **0.566** mAP50-95 evaluated at its
trained eps and **0.151** after a stock reload.

A checkpoint trained before 1.5.0 carries eps=1e-5 semantics. To report
faithful numbers for it, either evaluate with BN eps overridden to 1e-5:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

or fold `sqrt((var + 1e-3) / (var + 1e-5))` into the BN weights once and save
the result. Checkpoints trained on 1.5.0 and later need neither.

### D-FINE multi-scale training uses the upstream per-size recipe

`base_size_repeat` was hardcoded to 3 for every size. It now resolves per size
as upstream specifies: **n** trains at fixed size with multi-scale off, **s**
20, **m** 6, **l** 4, **x** 3. Only x matched before, so n, s, m and l see a
different scale distribution and converge to different metrics.

To restore the old behavior, set it explicitly:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM still uses the hardcoded 3. Family details are on
[D-FINE](/docs/models/d-fine).

## Worth knowing, no action needed

- **Rectangular `imgsz` results changed because they were wrong before.** Box
  coordinates, RTMDet mask resizing, YOLO-NAS rescaling and validator
  ground-truth scaling now use per-axis height and width instead of one scalar.
  Square `imgsz` is bit-unchanged. Rectangular inference or validation run on
  1.4.0 was mis-scaled. YOLO-NAS now rejects rectangular `imgsz` outright
  rather than silently producing wrong output.
- **Metrics dictionaries gained keys.** `max_det`, `ar_max_det` and
  `AR_max_det` from the COCO evaluator, and `metrics/loss` plus
  `metrics/loss/ce` from FOMO. Values at defaults are unchanged, but anything
  iterating metric keys, including custom
  [loggers](/docs/train/loggers) and CSV headers, sees new columns.
- **Seeded YOLO9 runs that trigger a head rebuild** start from a different
  initialization, because the seed is now applied before the rebuild rather
  than after. A seeded 1.4.0 fine-tune onto a different class count is not
  reproducible bit-for-bit on 1.5.0.
- **`libreyolo[hub-kernels]` on CUDA now actually engages the native
  MS-deform-attn kernel.** 1.4.0 gated it behind a condition RF-DETR never
  took, so the kernel never ran. Predictions can shift at float tolerance for
  RF-DETR and the other deformable-attention families. Stock installs are
  unaffected, and `LIBREYOLO_HUB_KERNELS=0` disables it.
- **`libreyolo predict` drops unsupported options instead of raising.** The CLI
  filters kwargs against the model's `__call__` signature, so an option a
  family does not accept is ignored rather than raising `TypeError`. A typo in
  a flag name is now silently ignored.
- **Live sources change the JSON output shape.** Webcams, RTSP streams and
  screen capture implicitly enable streaming, which emits one record per frame
  rather than one for the call. These
  [sources](/docs/predict/sources) are new in 1.5.0, so no 1.4.0 script is
  affected.
- **Re-exporting `rfdetr-pose` or `yolonas-pose` to ONNX yields different
  output names.** 1.4.0 misread their multi-tensor pose heads as segmentation
  through an output-count heuristic. Existing `.onnx` files on disk are
  untouched.
- **On a torch-free install**, results hold numpy arrays rather than
  `torch.Tensor`, so `.boxes.data` returns a different type and NMS
  tie-breaking may differ from torchvision. With torch installed, behavior is
  byte-for-byte unchanged. See
  [lightweight install](/docs/lightweight-install).
- **Config objects validate more at construction.** `TrainConfig` gained a
  `__post_init__` where it had none, so a config that was already invalid now
  raises immediately instead of failing deep into a run. `ValidationConfig`
  serialization gained an `edge_thresholds` key, which breaks a strict
  `ValidationConfig(**dump)` round-trip from a 1.4.0 dump.
- **Weight filenames for task-suffixed families resolve differently.**
  `segformer-b0` now resolves to `LibreSegformerb0-sem.pt`. This fixes
  auto-download 404s, and breaks any script that hardcoded the old unsuffixed
  filename.
- **The pytest marker `experimental_backend` is now `extended_backend`.** Only
  relevant if you run the test suite with `-m`.

## Checkpoints and datasets

Checkpoints written by 1.4.0 load unchanged. The
[schema](/docs/reference/checkpoint-schema) gained `imgsz_h` and `imgsz_w` for
rectangular models, and still writes the scalar `imgsz = max(h, w)` for older
readers. [ExecuTorch](/docs/export/executorch) and [MNN](/docs/export/mnn)
exports now require a sidecar, `<program>.pte.json` and `<model>.mnn.json`
respectively, and HRNet exports carry `pose_input: "person_crop"`. Dataset
formats are unchanged.
