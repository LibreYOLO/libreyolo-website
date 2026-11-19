---
title: Augmentations
seo_title: "Training augmentations in LibreYOLO, with examples"
description: "What each LibreYOLO augmentation knob does, shown on a real image by the library's own training code, with its range, its default and the families that read it."
lead: "Training augmentations are random transforms applied to every training sample. They are configured by knobs on TrainConfig, and each model family runs its own pipeline, so a knob changes samples only where that pipeline reads it."
keywords:
  - yolo data augmentation
  - mosaic augmentation
  - mixup
  - hsv augmentation
  - random affine
  - random erasing
  - small object detection
  - augmentation examples
  - copy paste augmentation
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: "1.8.0"
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
  small:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # YOLOX starts from mosaic_scale=(0.1, 2.0), which can shrink an
        # object to a tenth of its size. (0.5, 1.5) stops at half.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: |
        # A pair on the CLI is two comma-separated numbers.
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
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
---

## Setting the knobs

The augmentation knobs are ordinary `train()` arguments.

<code-tabs name="train" />

Three of them have shorter spellings, accepted by `train()` and the CLI alike:
`mosaic` maps to `mosaic_prob`, `fliplr` to `flip_prob`, and on detection
`mixup` maps to `mixup_prob`. Passing a short spelling and its field with
different values raises. `perspective` has no CLI flag in 1.8.0, so it is set
from Python.

The classification knobs are validated when their pipeline is built, and the
CLI also checks `flip_prob`, `flipud`, `cutmix` and `erasing`. `TrainConfig`
does not range-check the detection knobs, so `degrees=100` trains with
rotations of up to 100 degrees. The [augmentation
matrix](/docs/reference/augmentation-matrix) gives the range of every knob.

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

Called from Python, `train()` does not warn about ignored knobs. The [augmentation
matrix](/docs/reference/augmentation-matrix) has the state of every knob for
every family and the value each family starts from. The table is the shipped
spec at `libreyolo/data/augment/spec.py`, which the library's own tests check
against the pipelines, and it can be read directly:

<code-tabs name="support" />

## What each knob does

The figures below are produced by LibreYOLO 1.8.0's own training transforms,
run on the sample image through a one-image dataset, so every mosaic tile and
every MixUp partner is the same photo. The box is the person that LibreYOLO9t
detects on that image. Under each panel are the setting and the size of the
largest box, in pixels at the 640-pixel training resolution. Without
augmentation the person measures 291 × 217 px.

Several geometric knobs are ranges: `degrees=10` means an angle drawn anywhere
from -10 to +10 degrees. Their panels pin the draw at the value in the caption,
so they show the strongest effect a setting allows rather than a typical
sample. The detection figures come from the YOLOv9 pipeline. YOLOv7 trains
through it unchanged, and RT-DETR and RT-DETRv2 use its mosaic wrapper with a
per-sample transform of their own. YOLOX, RTMDet, PicoDet and FOMO build the
mosaic and the warp the same way in a separate wrapper, whose MixUp differs.

### mosaic_prob

The probability that a sample is a mosaic: four images placed on a canvas twice
the input size, which the affine warp then maps down to the input size. On the
YOLOX-style pipelines the warp and MixUp run only on mosaic samples, so both
stop when this is 0. RT-DETR, RT-DETRv2, PicoDet and FOMO ship with it at 0.

<aug-figure name="mosaic" />

The second panel pins the warp's scale at 0.5×, where the whole canvas fits.
The last two are ordinary draws with YOLOv9's defaults, and the caption gives
the scale each one drew.

### mosaic_scale

The range the warp's zoom is drawn from. At 1× an object keeps the size it has
at inference, and at 0.5× it is half that. The base default is `(0.1, 2.0)`,
which YOLOX, YOLOv7, RTMDet, PicoDet, RT-DETR, RT-DETRv2 and FOMO keep, so
their objects can shrink to a tenth of their size. YOLOv9 ships with
`(0.5, 1.5)`. YOLO-NAS and PP-YOLOE have no mosaic and reuse this range as the
scale of their per-sample affine warp.

<aug-figure name="mosaic_scale" />

Above 1× the output shows only the top-left part of the canvas, so objects are
cut at the edge or leave the frame.

### degrees

The rotation range: each warped sample turns by an angle drawn from
`-degrees` to `+degrees`. The rotation pivots on the top-left corner of the
canvas, so a large angle swings much of the image out of frame. YOLOX and
YOLOv7 keep the base default of 10, and YOLOv9 ships with 0.

<aug-figure name="degrees" />

### translate

The shift range, as a fraction of the input side: each warped sample moves by up
to `translate` times 640 px on each axis at `imgsz=640`. The base default is
0.1. YOLO-NAS and PP-YOLOE ship with 0.25, and RTMDet, PicoDet and FOMO with 0.

<aug-figure name="translate" />

### shear

The shear range in degrees, drawn separately for each axis. YOLOX and YOLOv7
keep the base default of 2, and every other family that reads it ships with 0.

<aug-figure name="shear" />

### perspective

A projective warp on top of the rotation, scale, shear and translation. Each
canvas corner moves inward by a random amount of up to
`min(perspective * 100, 0.2)` times half the canvas side, per axis, so
`perspective=0.001` moves a corner by at most 5 percent of the side at any
resolution, and every value from `0.002` up reaches the same maximum. The
default `0` keeps the warp affine, and a negative value raises `ValueError` when
the first sample is warped. The panels share one random draw, scaled by the
value. YOLOv9 and its E2E and P2 variants, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE,
RTMDet, PicoDet, RT-DETR and RT-DETRv2 apply it.

<aug-figure name="perspective" />

### hsv_prob

The probability of an HSV color jitter. Only the probability is a knob. The
jitter shifts hue by up to 5 on OpenCV's 0 to 179 hue scale and saturation and
value by up to 30 on 0 to 255, and each of the three shifts is skipped with
probability one half. Every detection pipeline that reads `hsv_prob` uses those
amounts. The last three panels are the most visible of 119 draws.

<aug-figure name="hsv" />

### flip_prob and flipud

The horizontal and vertical flip probabilities. `flip_prob` starts at 0.5 in
every family except FOMO, which ships with 0. `flipud` starts at 0. RTMDet,
PicoDet, RT-DETR, RT-DETRv2, FOMO, PP-YOLOE and the DETR-style families have no
vertical flip and ignore it. PP-YOLOE's recipe uses a random 90-degree rotation
instead.

<aug-figure name="flip" />

### mixup_prob and mixup_scale

MixUp blends a second sample into the current one and keeps the boxes of both.
In the YOLOv9 pipeline the partner is a whole sample without a mosaic, the blend
ratio is drawn uniformly between 0 and 1, and `mixup_scale` is never read,
although the library's spec lists it as gated by mosaic. In the YOLOX wrapper, used by
YOLOX, RTMDet, PicoDet and FOMO, and in the YOLO-NAS one, the partner is first
rescaled by a factor drawn from `mixup_scale` and the two images are averaged
at a fixed 0.5. The base default for `mixup_prob` is 1.0, and YOLOv9 ships with
0.

<aug-figure name="mixup" />

The last two panels blend a second sample into a mosaic at 0.5×. Their
caption gives the blend ratio λ, the weight of the mosaic sample.

### no_aug_epochs

`no_aug_epochs` is `used` everywhere, but it does not mean the same thing
everywhere. On the mosaic pipelines it turns mosaic and MixUp off for the final
epochs. On the DETR-style pipelines it stops the photometric, zoom-out and crop
augmentations and shapes the schedule's tail. On classification pipelines it
disables auto-augmentation, erasing, MixUp and CutMix, and crop and flips
remain.

## DETR-style recipes

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4 and TinyFormer train through one
torchvision transform with no mosaic and no affine warp, and so does EdgeCrafter
(EC) for detection. The transform's three strong operations are recipe
constants rather than knobs:

- `RandomPhotometricDistort` applies each color change (brightness 0.875 to
  1.125, contrast and saturation 0.5 to 1.5, hue ±0.05, and a channel
  shuffle) with probability 0.5.
- `RandomZoomOut`, with probability 0.5, places the image on a black canvas up
  to four times its side, so objects can shrink to a quarter of their size.
- `RandomIoUCrop`, applied with probability 0.8, crops to a region between 0.3
  and 1.0 of each side, chosen to keep at least part of the boxes.

After these steps, boxes 1 px or less on either side are dropped. All three operations
stop at the family's stop epoch, at the latest when the final `no_aug_epochs`
begin. The only knobs this pipeline reads are `flip_prob` and
`no_aug_epochs`.

<aug-figure name="detr_ops" />

Three samples from the start of a D-FINE run, with all three operations live:

<aug-figure name="detr_samples" />

RF-DETR has a pipeline of its own: a horizontal flip, then with probability
`crop_resize_prob` a resize of the shorter side to 400, 500 or 600 px and a
random square crop of 384 to 600 px, and finally a resize straight to the
square training resolution, which stretches rather than letterboxes.
`crop_resize_prob` defaults to 0.5 and is set from Python.

<aug-figure name="rfdetr_crop" />

## Small objects

`erasing` does not apply to detection. Random Erasing belongs to the
classification pack and no detection pipeline has an erasing step, so a
detector trains the same with any value of it, and there is nothing to turn
off. These are the steps that shrink or remove small objects:

- `mosaic_scale`. On the mosaic pipelines an object's size is multiplied by the
  drawn scale, so the lower bound sets the smallest it gets. Under the base
  default `(0.1, 2.0)` a 20 px object can come out at 2 px. The YOLOv9
  pipeline then drops boxes 2 px or less on either side, and its final
  transform, like YOLOX's, drops boxes 1 px or less on either side. The
  object's pixels stay in
  the image with no label. A lower bound of 0.5, as in YOLOv9's `(0.5, 1.5)`,
  keeps every object at half its inference size or more.
- `degrees`, `translate`, `shear` and `perspective` push objects to the edge,
  where their boxes are clipped and pass through the same size filter.
- On the DETR-style recipes, `RandomZoomOut` can shrink objects to a quarter of
  their size and `RandomIoUCrop` can cut them out. Neither is a knob, and both
  stop at the stop epoch.
- RF-DETR's crop can cut objects at its edge. `crop_resize_prob=0` turns it
  off.
- The training resolution scales every object: the letterbox resizes an image
  by `imgsz` divided by its longer side, so at `imgsz=1280` every object has
  twice the width and height it has at 640. [YOLOv9](/docs/models/yolov9) also
  has a P2 variant with a stride-4 head for small objects.

<code-tabs name="small" />

## The classification pack

Six knobs act only on the classification pipeline. Detection families ignore
all six.

<code-tabs name="classify" />

`scale` is the area range of the random crop. `scale=0.5` means `(0.5, 1.0)`,
and an explicit pair sets both bounds. A small lower bound can crop the object
out entirely, as in the third panel.

<aug-figure name="cls_crop" />

`auto_augment` takes `"randaugment"`, `"autoaugment"`, `"augmix"` or `None`,
and applies one draw of that policy to each sample.

<aug-figure name="auto_augment" />

`erasing` is the probability of Random Erasing, in `[0, 1)`. It runs after
normalization and fills a rectangle covering 2 to 33 percent of the image with
zeros, which after normalization is the ImageNet mean color.

<aug-figure name="erasing" />

`mixup` and `cutmix` are per-batch probabilities that produce soft labels. At
most one runs per batch, MixUp first, so the two are additive and must sum to
at most 1. In the panels, A and B are two crops of the same photo, and each
caption gives the soft label the mixed image trains on.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup` and `cutmix` default off, so
classification training is unchanged unless you ask.
`train()` and the CLI route `mixup` by task: classification uses batch mixing,
while detection uses `mixup_prob`.

`crop_pct=None` preserves the family's evaluation resize ratio. An override
changes the evaluation crop of the validation pass that runs during training,
while export keeps the family's native preprocessing.

The CLI exposes `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`,
`fliplr` and `flipud`. Classification `mixup` means batch mixing and defaults
to 0.0. Horizontal flips default to 0.5 and vertical flips to 0.0.

`no_aug_epochs` disables auto-augmentation, erasing, MixUp and CutMix for the
final epochs, retaining crop and flips. Classification recipes default this
tail to 0.

## Family-specific knobs

Some knobs live on a family's config subclass rather than on the base class, so
they exist for that family only and have no CLI flag.

| Family | Knob | Effect |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Copy-paste instance augmentation probability, `task="segment"` only |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` reuses the same sample mirrored, `"mixup"` pulls a second sample |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Random 90 degree rotation probability |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Per-image ground-truth cap in the train transforms, default 300 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste for `task="segment"`, `"flip"` mode only |
| RF-DETR | `crop_resize_prob` | Random crop-resize probability in the native pipeline, default 0.5 |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Random crop-resize probability, `task="segment"` only |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Pose-path jitter and keypoint-aware affine probabilities |

`max_labels` is the one that loses data. Boxes past the cap are dropped with a
one-time warning rather than an error, so dense imagery such as aerial
photography needs it raised. Label-assignment memory follows the densest image
in the batch, not the cap.

Mosaic and MixUp are disabled for oriented-box training regardless of the knobs,
because corner-aware augmentation for rotated boxes is not implemented.

## Related

- [Augmentation matrix](/docs/reference/augmentation-matrix) for every
  family's state for every knob, the ranges, and each family's defaults.
- [Hyperparameters](/docs/train/hyperparameters) for `no_aug_epochs` as a
  schedule argument and the rest of `train()`.
- [Datasets](/docs/train/datasets) for the label formats these transforms consume.
