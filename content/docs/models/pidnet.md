---
title: PIDNet
families: [pidnet]
seo_title: "PIDNet: predict and export real-time segmentation under MIT"
description: "Use PIDNet in LibreYOLO for real-time semantic segmentation. Install, predict, validate and export the s/m/l Cityscapes checkpoints under MIT."
lead: "A three-branch semantic segmentation network that adds a dedicated boundary branch to a proportional-integral-derivative-inspired design, aimed at real-time inference. LibreYOLO ships it for semantic segmentation only."
keywords: [PIDNet, real-time semantic segmentation, boundary-aware segmentation, Cityscapes, dense prediction]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) class ids
        print(mask.classes)      # sorted class ids present in the image
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePIDNets-sem.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibrePIDNets-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
---

## Install

PIDNet needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally. The
`-sem` filename suffix is required for this family.

<code-tabs name="predict" />

Semantic segmentation returns one class id per pixel, not boxes, so
`result.semantic_mask` carries a `(H, W)` array on `.data` and the list of
class ids present in the image on `.classes`. `conf`, `iou` and `max_det` are
accepted for API parity but have no effect: the model assigns a class to every
pixel by argmax, with no confidence threshold or NMS step. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Three sizes, all at a fixed 1024 px input. The published checkpoints are
conversions of the official PIDNet Cityscapes weights, 19 classes.

LibreYOLO does not train PIDNet: `train()` raises `NotImplementedError` for
this family, which the [support tier](/docs/models) above marks as inference
only.

## Validate

`val()` returns `metrics/mIoU` and `metrics/pixel_accuracy`, measured against
any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`. [Export](/docs/export) lists the arguments every format accepts.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
