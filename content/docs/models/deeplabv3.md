---
title: DeepLabv3
families: [deeplabv3]
seo_title: "DeepLabv3: predict and export ASPP segmentation under BSD-3-Clause"
description: "Use DeepLabv3 in LibreYOLO for semantic segmentation. Install, predict, validate and export torchvision's ResNet and MobileNetV3 checkpoints."
lead: "A semantic segmentation network that pools features at several dilation rates in parallel (atrous spatial pyramid pooling) before classifying each pixel. LibreYOLO ships it for semantic segmentation only."
keywords: [DeepLabv3, atrous spatial pyramid pooling, ASPP, semantic segmentation, dense prediction]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) class ids
        print(mask.classes)      # sorted class ids present in the image
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDeepLabv3r50-sem.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
---

## Install

DeepLabv3 needs no optional extra. Everything it imports is in the base
install.

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

Three backbones: dilated ResNet-50, dilated ResNet-101, and dilated
MobileNetV3-Large. This is DeepLabv3, not DeepLabv3+, so there is no decoder
stage or CRF refinement, matching torchvision's implementation rather than the
paper's own reference code.

LibreYOLO does not train DeepLabv3: `train()` raises `NotImplementedError` for
this family, which the [support tier](/docs/models) above marks as inference
only. The three published checkpoints are torchvision's own COCO-with-VOC-label
weights, converted for LibreYOLO's loader.

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
