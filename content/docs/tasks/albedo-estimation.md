---
title: Albedo estimation
seo_title: Albedo estimation | LibreYOLO
description: Albedo estimation predicts intrinsic surface color as linear RGB.
lead: Albedo estimation predicts intrinsic surface color as linear RGB.
keywords:
  - Albedo estimation
  - LibreYOLO
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Requires CUDA and libreyolo[marigold].
        model = LibreYOLO("LibreMarigoldV2b-albedo.pt", device="cuda")
        result = model(SAMPLE_IMAGE)
        print(result.albedo)
---

## Models

[Marigold V2](/docs/models/marigold-v2) supplies the albedo checkpoint. Its default four-bit runtime requires CUDA and the `marigold` extra.

## Predict

<code-tabs name="predict" />

`result.albedo` stores linear RGB. Plotting converts it to sRGB for display; use the linear values for numeric comparisons.

## Dataset format

Validation pairs each image in `images/<split>` with a same-stem floating-point `.npy` file in `albedo/<split>`. Each target holds `(H, W, 3)` linear RGB in [0, 1] at the same size as its image. Set `input_dir` and `albedo_dir` in the dataset YAML to rename those folders. See [dataset formats](/docs/reference/dataset-formats#albedo).

## Train

Marigold V2 does not support training through LibreYOLO.

## Validate

Validation reports PSNR and SSIM in linear RGB, with PSNR as fitness. It does not compare display-converted sRGB images.

## Export

Export is not supported.
