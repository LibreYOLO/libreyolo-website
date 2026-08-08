---
title: AlexNet
families: [alexnet]
seo_title: "AlexNet: run the classic ImageNet classifier in LibreYOLO"
description: "Predict, validate and export AlexNet with LibreYOLO. BSD-3-Clause torchvision weights; fine-tuning is not yet supported."
lead: "AlexNet is the convolutional network that won ILSVRC 2012 and helped start the deep learning era in computer vision. LibreYOLO ships the single-tower, later revision of the architecture for image classification."
keywords: [AlexNet, ImageNet, convolutional neural network, image classification]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreAlexNetb-cls.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data is a directory root with train/ and val/ class-folder splits
        # (ImageFolder layout), not a dataset YAML.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Install

AlexNet needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

A classifier returns `result.probs` instead of `result.boxes`: `top1`
and `top5` give class indices, `top1conf` and `top5conf` give their
confidences. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Variants

One size. The shipped graph is the later single-tower revision released by
torchvision, with 64 first-layer filters and no local response
normalization, not the original two-GPU 2012 architecture. LibreYOLO ships
this family inference-only: prediction, ImageNet-style top-1/top-5
validation and export are supported, and fine-tuning is not implemented.

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
