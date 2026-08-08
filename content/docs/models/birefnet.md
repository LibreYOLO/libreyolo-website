---
title: BiRefNet
families: [birefnet]
seo_title: "BiRefNet: background removal and matting in LibreYOLO"
description: "Use BiRefNet in LibreYOLO for background removal and dichotomous image segmentation. Install, predict, validate and export the general checkpoint."
lead: "A bilateral-reference network that predicts a soft alpha matte separating a subject from its background. LibreYOLO ships inference and validation for BiRefNet's matte task."
keywords: [BiRefNet, background removal, dichotomous image segmentation, alpha matte, image matting, cutout]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreBiRefNetl-matte.pt source=bus.jpg save=True
    - label: Cutout
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: source RGB plus the matte as an alpha channel.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # A directory containing images/ and an auto-detected matte directory
        # (mattes/, matte/, gt/, masks/, mask/ or alpha/) also works in place
        # of a dataset YAML.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
---

## Install

BiRefNet needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

A matte result carries no boxes; `result.matte` is a dense `(H, W)`
float32 array in `[0, 1]`, 1 fully foreground and 0 fully background. Unlike a
binary mask, the soft matte keeps anti-aliased edge detail such as hair and
fur. `result.cutout()` composites the source image with that alpha channel
into an RGBA array, and `result.save(path)` (or `save=True` on the predict
call) writes it straight to a transparent-background PNG. The model runs at a
fixed native 1024x1024 canvas; a different resolution is not supported,
because the Swin backbone's relative-position tables are tied to it, and a
mismatch interpolates them badly rather than raising an error. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

One published checkpoint, `l`, the Swin-L tier BiRefNet-general model and the
quality default upstream. The family's code also supports a Swin-T lite tier,
`t`, but no LibreYOLO conversion of it is published yet.

## Validate

`val()` reports two metrics over a paired image/matte folder, both in
`[0, 1]` and independent of resolution: MAE, the mean absolute error against
the ground-truth alpha (lower is better), and S-measure (Fan et al., ICCV
2017), a structural similarity that credits preserving the subject's shape and
holes, which pixel MAE alone misses (higher is better). Validation drives the
model's own `predict`, so it uses the family's exact preprocessing.

<code-tabs name="val" />

Validation is inference-only; fine-tuning is a documented follow-up rather
than a shipped feature (see Predict for the exact resolution constraint that
any future trainer would inherit).

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` file behaves like a checkpoint and returns the same `Results`.
TorchScript is the validated path; ONNX conversion runs but has not cleared
the same parity bar. [Export](/docs/export) lists the arguments every format
accepts and the extras a few of them add.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
