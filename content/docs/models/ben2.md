---
title: BEN2
families:
  - ben2
seo_title: BEN2 in LibreYOLO
description: BEN2 predicts a soft foreground alpha matte for background removal.
lead: BEN2 predicts a soft foreground alpha matte for background removal.
keywords:
  - BEN2
  - LibreYOLO
  - matte
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBEN2b-matte.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.matte.array.shape)
        result.save("cutout.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBEN2b-matte.pt", device="cpu")
        # data: a matte dataset YAML, or a folder with images/ and mattes/ (grayscale alpha, same file stems)
        metrics = model.val(data="path/to/your/matte_dataset.yaml", workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBEN2b-matte.pt", device="cpu")
        model.export(format="onnx")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

Prediction uses a fixed 1024-pixel canvas. Native prediction accepts batches; exports require batch 1. `result.save()` writes a transparent cutout. Training is not supported.


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
