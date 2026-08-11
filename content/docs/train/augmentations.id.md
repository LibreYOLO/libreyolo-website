---
title: Augmentations
seo_title: Training augmentations in LibreYOLO
description: >-
  The augmentation knobs on TrainConfig, the four pipeline shapes behind them,
  and the per-family table saying which knobs are used, gated or ignored.
lead: >-
  Augmentation is configured by knobs on TrainConfig, but each model family runs
  its own training pipeline, and a pipeline that has no mosaic branch ignores
  mosaic_prob rather than approximating it.
keywords:
  - yolo data augmentation
  - mosaic augmentation
  - mixup
  - hsv jitter
  - random affine
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # The CLI spells mosaic_prob as mosaic and mixup_prob as mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: Read the support table for a family
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Just the ignored ones
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Classification pack
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## Setting the knobs

The augmentation knobs are ordinary `train()` arguments.

<code-tabs name="train" />

Two of them have shorter CLI spellings: `mosaic` maps to `mosaic_prob` and
`mixup` maps to `mixup_prob`. Every other knob is spelled identically in both
places.

## Three states, not two

Whether a knob does anything depends on the family. The library keeps a
declarative table of that, and each entry is one of three states.

`used` means the knob reaches the pipeline and changes samples. `ignored` means
it never reaches the pipeline, so setting it does nothing. `gated_by_mosaic`
means it only applies to samples that took the mosaic branch, so with
`mosaic_prob=0` it never fires even though it is wired up.

That third state is the one that surprises people. On a YOLOX-style pipeline the
affine warp runs on the mosaic canvas and MixUp blends a mosaic sample, so
`mosaic_prob=0` silently disables `degrees`, `translate`, `shear`,
`perspective`, `mosaic_scale`, `mixup_prob` and `mixup_scale` all at once. The
trainer logs a warning for the MixUp case specifically:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

The CLI warns about ignored knobs too, listing only the ones you actually typed:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Four pipeline shapes

Families cluster into four training pipelines, and the pipeline determines almost
all of the answers.

The YOLOX-style mosaic pipeline applies HSV jitter and flips per sample, then
runs affine and MixUp inside the mosaic branch. It covers YOLOX, YOLOv7, YOLOv9
and its E2E and P2 variants, RTMDet, PicoDet, RT-DETR, RT-DETRv2 and FOMO.

The DETR-style pass-through pipeline has no mosaic and no affine warp. Its
photometric distortion, zoom-out and IoU crop are recipe constants rather than
config knobs, so only `flip_prob` and `no_aug_epochs` are live. It covers D-FINE,
Dome-DETR, DEIM, DEIMv2, RT-DETRv4, EC and, with one change, RF-DETR.

The classification ImageFolder pipeline ignores every detection knob. Its
horizontal flip is a fixed 0.5 that `flip_prob` does not reach. It has its own
knob pack instead, described below.

YOLO-NAS is a shape of its own: no mosaic at all, an always-on per-sample affine,
and MixUp applied independently rather than gated. Its `mosaic_scale` value is
reused as the affine scale range.

SegFormer and NAFNet each run a task-specific pipeline whose randomness is fixed
in the family rather than configurable. For SegFormer the live knobs are the
class attributes `semantic_scale_jitter` and `semantic_hsv_prob`, not
`mosaic_scale` and `hsv_prob`. NAFNet's crop and flips are coupled input and
target operations at a fixed 0.5 probability.

## Which family honors which knob

The table below is the shipped spec at
`libreyolo/data/augment/spec.py`, which is asserted against the real pipeline
plumbing by the library's own tests. Read it there rather than inferring from the
architecture.

<code-tabs name="support" />

Summarized by pipeline, for the base knobs:

| Knob | YOLOX-style | YOLO-NAS | DETR-style | Classification |
|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored |
| `mixup_prob` | gated by mosaic | used | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored |
| `flip_prob` | used | used | used | ignored |
| `flipud` | used | used | ignored | ignored |
| `degrees` | gated by mosaic | used | ignored | ignored |
| `translate` | gated by mosaic | used | ignored | ignored |
| `shear` | gated by mosaic | used | ignored | ignored |
| `perspective` | gated by mosaic | used | ignored | ignored |
| `mosaic_scale` | gated by mosaic | used | ignored | ignored |
| `mixup_scale` | gated by mosaic | used | ignored | ignored |
| `no_aug_epochs` | used | used | used | used |

Exceptions inside those columns, all of them narrowing:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2 and FOMO have no vertical flip, so
  `flipud` is ignored. FOMO's mosaic wrapper is also built without perspective.
- RF-DETR's native pipeline has no HSV jitter, so `hsv_prob` is ignored on top of
  the DETR-style column.
- EC honors `hsv_prob`, `degrees` and `translate`, but only for `task="pose"`,
  whose keypoint-aware transform reads them. Its detect and segment paths use
  fixed photometric recipes.
- DINOv2 follows the DETR-style column for its detect and semantic tasks and adds
  the classification pack for `task="classify"`.

`no_aug_epochs` is `used` everywhere, but it does not mean the same thing
everywhere. On the mosaic pipelines it turns mosaic and MixUp off for the final
epochs. On the DETR-style pipelines it stops the photometric, zoom-out and crop
augmentations and shapes the schedule's tail. On the classification and semantic
pipelines it only shapes the tail.

## The classification pack

Four knobs drive the classification pipeline and nothing else. Detection families
ignore all four.

<code-tabs name="classify" />

`auto_augment` takes `"randaugment"`, `"autoaugment"`, `"augmix"` or `None`.
`erasing` is the RandomErasing probability. `mixup` and `cutmix` are per-batch
probabilities producing soft labels; at most one runs per batch, MixUp first,
so the two are additive and should sum to at most 1.

All four default off, so classification training is unchanged unless you ask.

One naming collision is worth stating plainly: on the CLI, `mixup` is the alias
for the detection `mixup_prob`. The classification `mixup` field has no CLI
spelling of its own and is reachable only through `model.train(mixup=...)` in
Python.

## Family-specific knobs

Some knobs live on a family's config subclass rather than on the base class, so
they exist for that family only and have no CLI flag.

| Family | Knob | Effect |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Copy-paste instance augmentation probability, `task="segment"` only |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` reuses the same sample mirrored, `"mixup"` pulls a second sample |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Random 90 degree rotation probability |
| YOLOv9 | `max_labels` | Per-image ground-truth cap in the train transforms, default 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste for `task="segment"`, `"flip"` mode only |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Random crop-resize probability |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Pose-path jitter and keypoint-aware affine probabilities |

`max_labels` is the one that silently loses data. Boxes past the cap are dropped
without an error, so dense imagery such as aerial photography needs it raised.

Mosaic and MixUp are disabled for oriented-box training regardless of the knobs,
because corner-aware augmentation for rotated boxes is not implemented.

## Related

- [Hyperparameters](/docs/train/hyperparameters) for `no_aug_epochs` as a
  schedule argument and the rest of `train()`.
- [Datasets](/docs/train/datasets) for the label formats these transforms consume.


