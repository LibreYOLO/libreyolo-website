---
title: FCN
families: [fcn]
seo_title: "FCN: predict and export a ResNet FCN under BSD-3-Clause"
description: "Use FCN in LibreYOLO for semantic segmentation. Install, predict, validate and export torchvision's dilated-ResNet FCN checkpoints."
lead: "A dense per-pixel classifier that replaces a detector's fully connected layers with convolutions, so it outputs a full-resolution class map instead of boxes. LibreYOLO ships it for semantic segmentation only."
keywords: [FCN, fully convolutional network, semantic segmentation, dense prediction, ResNet]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) class ids
        print(mask.classes)      # sorted class ids present in the image
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFCNr50.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreFCNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
---

## Install

FCN needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

Semantic segmentation returns one class id per pixel, not boxes, so
`result.semantic_mask` carries a `(H, W)` array on `.data` and the list of
class ids present in the image on `.classes`. `conf`, `iou` and `max_det` are
accepted for API parity but have no effect: the model assigns a class to every
pixel by argmax, with no confidence threshold or NMS step. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Two ResNet depths, both at a fixed 520 px input. The library's inference graph
is torchvision's dilated-ResNet FCN, not the original paper's VGG-based FCN-8s
network with skip connections.

LibreYOLO does not train FCN: `train()` raises `NotImplementedError` for this
family, which the [support tier](/docs/models) above marks as inference only.
The two published checkpoints are torchvision's own COCO-trained weights,
converted for LibreYOLO's loader.

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

The authors publish no BibTeX block for this work. Cite the paper linked in
the Upstream row above.
