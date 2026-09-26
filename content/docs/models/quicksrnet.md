---
title: QuickSRNet
families:
  - quicksrnet
seo_title: QuickSRNet in LibreYOLO
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
        result.restored.save("upscaled.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreQuickSRNetm2-restore.pt", device="cpu")
        # Paired super-resolution dataset YAML: inputs/val low-res images, targets/val 2x high-res images
        metrics = model.val(data="path/to/your/restore.yaml", workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreQuickSRNetm2-restore.pt", device="cpu")
        model.export(format="onnx")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The Medium 2x model preserves native input resolution and doubles each spatial dimension. Validation uses paired low/high-resolution images and reports PSNR and SSIM. ONNX supports dynamic spatial shapes; TorchScript uses a fixed canvas. Training is not supported.


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Export

<export-matrix />

<code-tabs name="export" />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
