---
title: TinyFormer
families:
  - tinyformer
seo_title: TinyFormer in LibreYOLO
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
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreTinyFormers.pt", device="cpu")
        # coco8.yaml is a built-in 8-image detection dataset that downloads on first use.
        # Replace it with the path to your own detection dataset YAML.
        model.train(data="coco8.yaml", epochs=1, device="cpu", workers=0)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreTinyFormers.pt", device="cpu")
        # coco8.yaml downloads on first use; pass your own detection dataset YAML instead.
        metrics = model.val(data="coco8.yaml", workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreTinyFormers.pt", device="cpu")
        model.export(format="onnx")
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
