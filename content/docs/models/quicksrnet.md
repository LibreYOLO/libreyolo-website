---
title: QuickSRNet
families:
  - quicksrnet
seo_title: 'QuickSRNet: prediction and training in LibreYOLO'
description: QuickSRNet upsamples an image by a factor of two.
lead: QuickSRNet upsamples an image by a factor of two.
keywords:
  - QuickSRNet
  - LibreYOLO
  - restore
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreQuickSRNetm2-restore.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        result.save("upscaled.png")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The Medium 2x model preserves native input resolution and doubles each spatial dimension. Validation uses paired low/high-resolution images and reports PSNR and SSIM. ONNX supports dynamic spatial shapes; TorchScript uses a fixed canvas. Training is not supported.

## Export

<export-matrix />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
