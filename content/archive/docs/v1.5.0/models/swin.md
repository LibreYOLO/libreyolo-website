---
title: Swin Transformer
families: [swin]
seo_title: "Swin Transformer: classify images with LibreYOLO's LibreSwin"
description: "Predict, validate and export Swin Transformer classifiers with LibreYOLO. MIT weights; fine-tuning is not yet supported."
lead: "Swin Transformer V1: a hierarchical vision transformer that computes attention inside shifted local windows instead of over the whole image. LibreYOLO ships four sizes for image classification."
keywords: [Swin Transformer, hierarchical vision transformer, shifted window attention, image classification]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSwint-cls.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data is a directory root with train/ and val/ class-folder splits
        # (ImageFolder layout), not a dataset YAML.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreSwint-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Install

Swin needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

A classifier returns `result.probs` instead of `result.boxes`: `top1`
and `top5` give class indices, `top1conf` and `top5conf` give their
confidences. Every size is fixed to a 224px input, because the final attention
stage is built for that resolution; predict, validate and export all raise if
you pass a different `imgsz`. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Variants

Four sizes, tiny through large, built from the same shifted-window tower and
differing in embedding width and stage depth. Large is pretrained on
ImageNet-22k and fine-tuned on ImageNet-1k; the other three are trained on
ImageNet-1k directly. LibreYOLO ships this family inference-only: prediction,
ImageNet-style top-1/top-5 validation and export are supported, and the
upstream ImageNet training recipe is not implemented.

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
