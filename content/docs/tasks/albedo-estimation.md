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

[Marigold V2](/docs/models/marigold-v2) supplies the albedo adapters. Their default four-bit runtime requires CUDA and the `marigold` extra.

## Predict

<code-tabs name="predict" />

`result.albedo` stores linear RGB. Plotting converts it to sRGB for display; use the linear values for numeric comparisons.

## Dataset format

Validation takes paired input images and linear-RGB albedo targets through the albedo dataset loader. See [dataset formats](/docs/reference/dataset-formats).

## Train

Marigold V2 does not support training through LibreYOLO.

## Validate

Validation reports PSNR and SSIM in linear RGB, with PSNR as fitness. It does not compare display-converted sRGB images.

## Export

Export is not supported.
