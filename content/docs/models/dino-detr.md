---
title: DINO-DETR
families: [dinodetr]
seo_title: "DINO-DETR: predict and export under Apache-2.0"
description: "Run DINO-DETR in LibreYOLO for object detection. Install, predict, validate and export three denoising-anchor sizes, all Apache-2.0 licensed."
lead: "DINO-DETR, published by IDEA Research as DINO, combines contrastive denoising training with mixed query selection on top of Deformable DETR's sparse attention. LibreYOLO ships three sizes for detection, inference only."
keywords: [DINO-DETR, DINO, detection transformer, denoising anchor boxes, mixed query selection, object detection, IDEA Research]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDINODETRr50.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() returns a plain dict, not an object
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800
        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800 half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreDINODETRr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

DINO-DETR needs no optional extra. Everything it imports is in the base
install, using the same pure-PyTorch multi-scale deformable attention core as
LibreYOLO's Deformable DETR family.

```bash
pip install libreyolo
```

Installing `libreyolo[hub-kernels]` is optional. Once the `kernels` package is
present, LibreYOLO fetches a compiled multi-scale deformable attention kernel
from the Hugging Face Hub at runtime and uses it in place of the pure-PyTorch
core; `LIBREYOLO_HUB_KERNELS=0` turns it back off.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` and `max_det` filter the query
selection; `iou` is accepted for API parity but has no effect, because the
decoder is a set predictor with no NMS step. See
[prediction](/docs/predict) for sources, streaming and result handling.

DINO-DETR is inference-only in LibreYOLO. Upstream trains with contrastive
denoising and Hungarian matching; that recipe is not implemented here, so
`train()` raises `NotImplementedError`.

## Variants

Three checkpoints, all at the same input resolution. `r50` and `r50s5` share a
ResNet-50 backbone and differ in how many feature-map scales feed the decoder,
four against five. `swinl` swaps the backbone for Swin-L and also samples five
scales.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

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

<provenance-box>

The three official checkpoints come from the authors' Google Drive release
folder, not a Hugging Face model card. The upstream repository declares
Apache-2.0 at the repository level but does not attach a license file or
license metadata to the checkpoints themselves, so the redistribution basis is
that repository-level declaration rather than a checkpoint-specific grant.
Every LibreYOLO mirror ships the verbatim upstream Apache-2.0 license text
alongside a notice explaining this.

</provenance-box>

## Citation

<citation-block />
