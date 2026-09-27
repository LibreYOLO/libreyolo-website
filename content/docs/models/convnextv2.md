---
title: ConvNeXt V2
families:
  - convnextv2
seo_title: ConvNeXt V2 in LibreYOLO
description: ConvNeXt V2 is an image classifier with global response normalization.
lead: ConvNeXt V2 is an image classifier with global response normalization.
keywords:
  - ConvNeXt V2
  - LibreYOLO
  - classify
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtV2atto-cls.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.probs)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtV2atto-cls.pt", device="cpu")
        # "smoke10" auto-downloads; for your own data, pass a folder with train/ and val/ class subfolders.
        model.train(data="smoke10", epochs=1, device="cpu", workers=0)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtV2atto-cls.pt", device="cpu")
        metrics = model.val(data="smoke10", workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtV2atto-cls.pt", device="cpu")
        model.export(format="onnx")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The eight sizes run classification at 224 pixels. Official pretrained weights are CC-BY-NC-4.0.

## Train

Training rebuilds the classification head for an ImageFolder dataset. `cls_pw` accepts values from 0 to 1 and defaults to 0. `class_weights=True` selects the alternative sample-normalized weighting; it cannot combine with `cls_pw>0`. Weighting settings must match when resuming.

[Dataset setup](/docs/train/datasets) describes the required training data.

<code-tabs name="train" />


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

## Citation

<citation-block />
