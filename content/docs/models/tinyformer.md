---
title: TinyFormer
families:
  - tinyformer
seo_title: 'TinyFormer: prediction and training in LibreYOLO'
description: TinyFormer is a trainable transformer detector with a DINOv3 backbone.
lead: TinyFormer is a trainable transformer detector with a DINOv3 backbone.
keywords:
  - TinyFormer
  - LibreYOLO
  - detect
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreTinyFormers.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreTinyFormers.pt", device="cpu")

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

Choose a published checkpoint below. The naming distinguishes the VisDrone variants and Objects365-to-COCO variants. Prediction requires a square `imgsz`; rectangular inputs raise an error.

## Train

Training accepts a detection dataset YAML. Omitted `epochs`, `batch`, `imgsz`, `lr0` and `amp` use the family recipe.

[Dataset setup](/docs/train/datasets) describes the required training data.

<code-tabs name="train" />

## Export

<export-matrix />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
