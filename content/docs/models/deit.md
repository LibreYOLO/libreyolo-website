---
title: DeiT
families: [deit]
seo_title: "DeiT image classifier: predict, validate, export"
description: "Run DeiT image classifiers in LibreYOLO: a frozen, inference-only museum family in tiny, small and base sizes, under Apache-2.0."
lead: "DeiT (Data-efficient image Transformer) is a plain Vision Transformer classifier trained on ImageNet-1k alone, with no extra pretraining data. LibreYOLO carries the tiny, small and base patch-16 sizes as a frozen, inference-only exhibit."
keywords: [DeiT, Vision Transformer, ViT, image classification, ImageNet, data-efficient training, museum family]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeiTb-cls.pt source=bus.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
---

## Install

DeiT needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

This family is inference-only: `train()` raises `NotImplementedError`, so this
page has no Train section. Predict, validate and export are all supported.
Weights download from Hugging Face on first use and are cached locally. The
`-cls` suffix in the filename is required and selects the classification task.

<code-tabs name="predict" />

The returned `Results` object carries a `probs` tensor instead of `boxes`;
`top1` and `top5` index the 1,000 ImageNet-1k classes and `top1conf` is the
softmax score for the top prediction. Each size has a fixed input resolution
from its positional embedding: preprocessing resizes and center-crops to it,
and passing a different `imgsz` raises rather than silently resampling. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Validate

`val()` returns a dictionary with top-1 and top-5 accuracy, measured against a
dataset laid out in the conventional `train/<class>/` and `val/<class>/`
folder structure.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. Running the graph in a bare runtime, with no LibreYOLO installed, is
also supported, but then preprocessing and postprocessing are yours to write.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
