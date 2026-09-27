---
title: Augmentation matrix
seo_title: "Which LibreYOLO family honors which augmentation"
description: "Per-family augmentation knob support: the sixteen TrainConfig knobs, the three statuses, the six pipeline archetypes, and the knobs a family silently ignores."
lead: "Setting an augmentation knob does not guarantee it reaches the pipeline. This page records how each trainable family treats each knob on TrainConfig, using the declarative table the library ships as its single source of truth."
keywords:
  - libreyolo augmentation
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - augmentation support matrix
  - TrainConfig knobs
last_verified: "1.5.0"
verification: "Knob list, statuses, archetypes, per-family deviations and helper functions read from libreyolo/data/augment/spec.py at v1.5.0. That table is pinned to the real pipelines by tests/unit/test_augment_spec.py."
snippets:
  usage:
    - label: Ask the spec directly
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
---

## The knobs

These are `TrainConfig` field names, not CLI spellings. The CLI maps its own
aliases onto them, so `--mosaic` sets `mosaic_prob`.

| Knob | Meaning |
|---|---|
| `mosaic_prob` | Probability of building a 4-image mosaic sample |
| `mixup_prob` | Probability of blending in a second sample |
| `hsv_prob` | Probability of HSV color jitter |
| `flip_prob` | Horizontal-flip probability |
| `degrees` | Random-rotation range for the affine warp, in degrees |
| `translate` | Random-translation fraction for the affine warp |
| `mosaic_scale` | Random-scale range for the affine warp |
| `mixup_scale` | Jitter-scale range applied to the MixUp partner image |
| `shear` | Random-shear range for the affine warp, in degrees |
| `perspective` | Projective warp magnitude for the affine warp |
| `flipud` | Vertical-flip probability |
| `no_aug_epochs` | Final epochs trained with strong augmentation disabled |
| `auto_augment` | Classification AutoAugment policy: randaugment, autoaugment or augmix |
| `erasing` | Classification RandomErasing probability |
| `mixup` | Classification batch-MixUp probability, with soft labels |
| `cutmix` | Classification batch-CutMix probability, with soft labels |

The last four are the classification pack. Detection families ignore them.
`mixup` is an API-only knob: the CLI `--mixup` is the alias for the detection
`mixup_prob`.

<code-tabs name="usage" />

## The three statuses

| Status | Meaning |
|---|---|
| `used` | The knob reaches the family's train pipeline and changes samples |
| `gated_by_mosaic` | The knob applies only to samples that took the mosaic branch, so with `mosaic_prob == 0` it never fires |
| `ignored` | The knob never reaches the pipeline; setting it does nothing |

`ignored` is the one worth checking before a run, because nothing fails. The
CLI warns when an explicitly set training parameter is one the selected family
ignores, and the trainer warns when `mixup_prob > 0` cannot fire because the
family gates MixUp on mosaic and `mosaic_prob` is zero.

## Pipeline archetypes

Every covered family follows one of six pipelines, with a handful of per-family
deviations listed below.

| Knob | YOLOX-style | YOLO-NAS | DETR-style | Classification | Semantic | Restore |
|---|---|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored | ignored | ignored |
| `mixup_prob` | gated | used | ignored | ignored | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored | ignored | ignored |
| `flip_prob` | used | used | used | ignored | ignored | ignored |
| `degrees` | gated | used | ignored | ignored | ignored | ignored |
| `translate` | gated | used | ignored | ignored | ignored | ignored |
| `mosaic_scale` | gated | used | ignored | ignored | ignored | ignored |
| `mixup_scale` | gated | used | ignored | ignored | ignored | ignored |
| `shear` | gated | used | ignored | ignored | ignored | ignored |
| `perspective` | gated | used | ignored | ignored | ignored | ignored |
| `flipud` | used | used | ignored | ignored | ignored | ignored |
| `no_aug_epochs` | used | used | used | used | used | used |
| `auto_augment` | ignored | ignored | ignored | used | ignored | ignored |
| `erasing` | ignored | ignored | ignored | used | ignored | ignored |
| `mixup` | ignored | ignored | ignored | used | ignored | ignored |
| `cutmix` | ignored | ignored | ignored | used | ignored | ignored |

In the YOLOX-style pipeline the per-sample preprocessing applies HSV jitter and
flips, while the affine warp and MixUp run only inside the mosaic branch.
YOLO-NAS instead runs a per-sample affine that is always on, ignores mosaic,
and applies MixUp independently, reusing `mosaic_scale` as the affine scale
range.

The DETR-style pipeline is a pass-through transform with no mosaic. Its
photometric distortion, zoom-out and IoU-crop are recipe constants rather than
configurable knobs, which is why `hsv_prob` and the geometry knobs never reach
it. The classification pipeline uses an ImageFolder transform whose horizontal
flip is a fixed 0.5 rather than `flip_prob`. Semantic scale jitter and HSV come
from family class attributes rather than config knobs, and restoration flips
are coupled input-and-target operations with a fixed 0.5 probability.

`no_aug_epochs` is honored everywhere, though what it turns off differs: mosaic
and MixUp for YOLOX-style, the affine and MixUp for YOLO-NAS, the strong
photometric and crop augmentations plus the learning-rate tail for DETR-style,
and the scheduler tail for the rest.

## Families by archetype

| Archetype | Families |
|---|---|
| YOLOX-style | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| DETR-style | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Classification | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Semantic | `segformer` |
| Restore | `nafnet` |

Twenty-five families are covered. A family outside this list returns an empty
ignored set, so no warning is emitted for it.

## Deviations

| Family | Difference from its archetype |
|---|---|
| `rtmdet` | `flipud` ignored: its transform has no vertical flip |
| `picodet` | `flipud` ignored |
| `rtdetr` | `flipud` ignored |
| `rtdetrv2` | `flipud` ignored |
| `fomo` | `perspective` and `flipud` ignored |
| `ec` | `hsv_prob`, `degrees` and `translate` used, for `task="pose"` only; detect and segment use fixed photometric recipes |
| `dinov2` | The classification pack is used, for `task="classify"` only |

`ec` and `dinov2` are multi-task families, so a knob is marked ignored only
when every one of the family's trainable tasks ignores it. That keeps the CLI
warning from ever being wrong for one task while right for another.

Dome-DETR inherits D-FINE's transforms unchanged. The one thing it cannot take
is multi-scale training, which its config disables rather than the
augmentation spec.

## Family-specific knobs

Some families carry augmentation knobs on their own `TrainConfig` subclass
rather than on the base. The CLI does not expose these; set them through the
Python API.

| Family | Knob | Meaning |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Copy-paste instance augmentation probability, `task="segment"` only |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Copy-paste source: `flip` mirrors the same sample, `mixup` uses a second sample |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Random 90-degree rotation probability |
| `rfdetr` | `copy_paste` | Copy-paste probability for `task="segment"`, `flip` mode only |
| `rfdetr` | `copy_paste_mode` | Copy-paste source mode for `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Random crop-resize probability in the native pipeline |
| `dfine` | `crop_resize_prob` | Random crop-resize probability, `task="segment"` |
| `ec` | `crop_resize_prob` | Random crop-resize probability, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Brightness and contrast jitter probability, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Keypoint-aware affine probability, `task="pose"` |

`rot90` applies to detect and OBB on `yolo9`.

## Querying the spec

| Helper | Returns |
|---|---|
| `aug_support(family)` | The knob-to-`Support` table, or `None` for an unknown family |
| `ignored_aug_params(family)` | The set of knob names the family ignores; empty for an unknown family |
| `uses_mosaic_gating(family)` | Whether the family's MixUp only fires on mosaic samples |
| `display_name(family)` | The human-facing family name used in warnings |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | The warning text when MixUp can never fire, else `None` |

A `Support` is a named tuple of `status` and `note`, where the note explains
why a knob is ignored or gated for that family.

## The mosaic gate

For a YOLOX-style family, `mixup_prob=0.5` with `mosaic_prob=0` disables MixUp
entirely, because MixUp applies only to mosaic samples. That combination is
easy to reach when turning mosaic off late in training. The trainer logs a
warning naming the family, and `mixup_gating_warning` is the pure function
behind it.
