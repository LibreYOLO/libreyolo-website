---
title: Background removal
seo_title: "Background removal in LibreYOLO"
description: "Cut a subject out of its background in LibreYOLO. Predict a soft alpha matte, write a transparent PNG, and validate with MAE and S-measure."
lead: "Background removal separates a subject from everything behind it. LibreYOLO exposes it as the matte task, which returns a soft alpha value per pixel rather than a hard foreground mask."
keywords: [background removal python, alpha matting model, dichotomous image segmentation, transparent png cutout, soft alpha matte]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Predict a matte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 in [0, 1]
    - label: Write a transparent PNG
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() composites the source with the matte as an alpha channel.
        result.save("subject.png")

        rgba = result.cutout()   # the same (H, W, 4) uint8 array in memory
        print(rgba.shape)
    - label: Composite onto a new background
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        rgba = result.cutout()
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        backdrop = np.full_like(rgba[..., :3], 255)          # white
        composited = (rgba[..., :3] * alpha + backdrop * (1 - alpha)).astype(np.uint8)
        print(composited.shape)
  val:
    - label: Validate and read the metric keys
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # A directory holding images/ and a matte directory works in place of
        # a dataset YAML.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # lower is better
        print(metrics["metrics/Smeasure"])   # fitness, higher is better
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Run the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
---

## Definition

The `matte` task predicts one alpha value per pixel from a single RGB image: `1`
is fully foreground and `0` is fully background. The value is continuous rather
than binary, which is the point of the task. A hard mask is one threshold away,
at 0.5, while the soft matte additionally carries the partial coverage at hair,
fur and motion-blurred edges that a binary mask throws away.

A prediction fills `result.matte`, a `Matte` payload holding an `(H, W)`
float32 array in `[0, 1]` on the original image canvas, reachable as NumPy
through `.array`. `result.cutout()` composites the source image with that
alpha into an `(H, W, 4)` uint8 RGBA array, and `result.save(path)` writes
the same thing to a transparent-background PNG. `result.boxes` stays empty,
so `conf`, `iou` and `max_det` have no effect.

## Models

Two families serve `matte`, and they share a forward path.

[BiRefNet](/docs/models/birefnet) is the bilateral-reference network the task is
built around, published here as one Swin-L tier checkpoint.

[FeyNobg](/docs/models/feynobg) is Feyn Inc.'s deepened variant: BiRefNet's
architecture with the third Swin stage grown from 18 to 24 blocks, then
retrained. LibreYOLO reuses BiRefNet's forward path, preprocessing and
single-logit output for it, so predict, validate and checkpoint handling behave
identically; the weights and the family identity are FeyNobg's own.

The two carry different weight licenses. Both are stated on the model pages, and
the license on the Hugging Face repository of the specific checkpoint is the
authoritative one.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

Both families run at a fixed native 1024x1024 canvas and resize the matte back
to the original image. A different resolution is not supported, because the Swin
backbone's relative-position tables are tied to that size, and a mismatch
interpolates them badly rather than raising. `Results.save()` is defined for
matte results only and needs the source image, which it reloads from
`Results.path` unless you pass one. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Dataset format

Matte validation pairs each RGB image with a single-channel ground-truth alpha
matte sharing the same stem, where 0 is background and 255 is foreground.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Passing that root as `data=` is enough: the matte directory is auto-detected
among `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` and `alpha/`. A dataset YAML
is the alternative, with `path` plus `val_images` and `val_mattes` naming
directories relative to it:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` and `names` are schema placeholders; a matte model returns `Results.matte`,
not detections. Matte values are read as alpha in `[0, 1]` by dividing by 255,
and a matte whose shape differs from the prediction canvas is resized bilinearly
to match. See [dataset formats](/docs/reference/dataset-formats) for the full
contract.

## Train

Neither matte family has a training implementation: `train()` raises
`NotImplementedError` on both, and matte support covers prediction, validation
and export only. Each model page names the upstream project that ships training
code and the conversion script that brings a checkpoint back.

## Validate

`val()` drives the model's own `predict`, so validation uses the family's exact
preprocessing, and both metrics are computed on the original image canvas.

<code-tabs name="val" />

`metrics/MAE` is the mean absolute error against the ground-truth alpha, in
`[0, 1]`, and lower is better. `metrics/Smeasure` is the S-measure of Fan et al.
(ICCV 2017), a structural similarity that credits getting the subject's shape
and its holes right, which a per-pixel average alone misses; higher is better.
S-measure is also `fitness`, the number best-checkpoint selection reads. Neither
metric depends on resolution.

## Export

An exported matte model loads back through `LibreYOLO()` on its file suffix, so
the artifact behaves like a checkpoint and returns the same `Results`.

<code-tabs name="export" />

TorchScript is the validated path for this task. ONNX conversion runs but has
not cleared the same parity bar, and the remaining formats are not available.
Per-format coverage is on the [BiRefNet](/docs/models/birefnet) and
[FeyNobg](/docs/models/feynobg) pages and in the
[full export matrix](/docs/reference/export-matrix).
