---
title: ViT
families: [vit]
seo_title: "ViT: run classic Vision Transformer classifiers in LibreYOLO"
description: "Predict, validate and export ViT classifiers with LibreYOLO. Apache-2.0 AugReg weights; fine-tuning is not yet supported."
lead: "The classic Vision Transformer: a pure transformer applied to fixed-size image patches, with a learned class token and no convolutions. LibreYOLO ships four AugReg-pretrained sizes for image classification."
keywords: [ViT, Vision Transformer, AugReg, image classification, transformer classifier]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        results = model(SAMPLE_IMAGE, save=True)

        probs = results[0].probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreViTti-cls.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data is a directory root with train/ and val/ class-folder splits
        # (ImageFolder layout), not a dataset YAML.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreViTti-cls.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].probs.top1)
---

## Install

ViT needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

A classifier returns `results[0].probs` instead of `results[0].boxes`: `top1`
and `top5` give class indices, `top1conf` and `top5conf` give their
confidences. Preprocessing resizes and center-crops to a fixed 224px input,
using timm's AugReg evaluation recipe: bicubic interpolation at a 0.9 crop
fraction. See [prediction](/docs/predict) for sources, streaming and result
handling.

## Variants

Four sizes, tiny through large, sharing one fixed 224px, patch-16 graph and
differing in embedding width and transformer depth. LibreYOLO ships this
family inference-only: prediction, ImageNet-style top-1/top-5 validation and
export are supported, and the AugReg fine-tuning recipe is not implemented.

## Validate

`val()` runs against an ImageFolder-style split (a directory with `train/` and
`val/` subfolders, one folder per class) and returns top-1 and top-5 accuracy.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. [Export](/docs/export) lists the arguments every format accepts and
the extras a few of them add.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
