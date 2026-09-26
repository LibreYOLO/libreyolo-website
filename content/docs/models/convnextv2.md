---
title: ConvNeXt V2
families:
  - convnextv2
seo_title: 'ConvNeXt V2: prediction and training in LibreYOLO'
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
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreConvNeXtV2atto-cls.pt", device="cpu")

        # Enter the path to your dataset YAML or classification folder.

        model.train(data=input("Dataset path: "), epochs=1, device="cpu",
        workers=0)
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

## Export

<export-matrix />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
