---
title: VGG
families: [vgg]
seo_title: "VGG: run VGG-16/19 image classifiers in LibreYOLO"
description: "Predict, validate and export VGG classifiers with LibreYOLO. BSD-3-Clause torchvision weights; fine-tuning is not yet supported."
lead: "VGG is a convolutional image classifier built from uniform stacks of small 3x3 convolutions instead of larger filters. LibreYOLO ships the 16- and 19-layer sizes, plain and with batch normalization, for image classification."
keywords: [VGG, VGG-16, VGG-19, convolutional neural network, image classification]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreVGG16-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data is a directory root with train/ and val/ class-folder splits
        # (ImageFolder layout), not a dataset YAML.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreVGG16-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Install

VGG needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

A classifier returns `result.probs` instead of `result.boxes`: `top1`
and `top5` give class indices, `top1conf` and `top5conf` give their
confidences. Prediction runs at a fixed 224px input and raises if you pass a
different `imgsz`. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Variants

Four sizes: 16 and 19 convolutional layers, each with a plain and a
batch-normalized variant. The shipped weights are torchvision's later
from-scratch ImageNet training, not conversions of the Oxford group's original
2014 Caffe release. LibreYOLO ships this family inference-only: prediction,
ImageNet-style top-1/top-5 validation and export are supported, and
fine-tuning is not implemented.

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

The authors publish no BibTeX block for this work. Cite the paper linked in
the Upstream row above.
