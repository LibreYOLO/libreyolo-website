---
title: Mask R-CNN
families: [mask_rcnn]
seo_title: "Mask R-CNN in LibreYOLO: predict, validate and export"
description: "Run Mask R-CNN in LibreYOLO for object detection and instance segmentation. Install, predict, validate and export the BSD-3-Clause torchvision port."
lead: "Mask R-CNN adds a per-region mask branch to Faster R-CNN, predicting a segmentation mask alongside each box it detects. LibreYOLO ports the torchvision implementation for detection and instance segmentation."
keywords: [Mask R-CNN, instance segmentation, object detection, Faster R-CNN, torchvision, two-stage detector]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMaskRCNNr50.pt source=bus.jpg save=True
    - label: Boxes only
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # task="detect" skips the mask head and returns boxes from the same
        # checkpoint, with no masks in the result.
        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # masks
        print(metrics["metrics/mAP50-95(B)"])   # boxes
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreMaskRCNNr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
---

## Install

Mask R-CNN needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in
a different detector is a one line change. Loading the checkpoint with no
`task` argument returns instance masks, since segmentation is this family's
default task; `result.masks` then carries them alongside the boxes.
Passing `task="detect"` loads the same weights without the mask head and
returns boxes only. `conf` and `iou` set the confidence and NMS thresholds;
Mask R-CNN keeps its upstream NMS step, unlike a query-based detector. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

One backbone: ResNet-50 with a feature pyramid, using torchvision's v2
Mask R-CNN builder. The published checkpoint carries a BSD-3-Clause license
and serves both tasks in this family, so there is no size to choose between.

## Validate

`val()` returns a dictionary of `metrics/` keys. Against this checkpoint's
default segmentation task, the plain `metrics/mAP50-95` key holds the mask
score, and the same run reports boxes under the `(B)` suffix so both are
available from one pass.

<code-tabs name="val" />

## Export

<export-matrix />

Mask R-CNN exports to ONNX only, at batch size 1. The exported graph keeps the
upstream resize and mask-paste steps inside it, so LibreYOLO forces
`dynamic=True` regardless of what is passed, to keep the graph valid for
sources that are not square. An exported `.onnx` file loads back through
`LibreYOLO()` on its file suffix and returns the same `Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family. The one checkpoint below is
listed under detect, but the same file loads for segmentation too: pass no
`task` argument and it returns masks by default.

<checkpoint-table />

## Licensing

<provenance-box>

Mask R-CNN is built as a subclass of LibreYOLO's Faster R-CNN wrapper: it
shares the same torchvision source and BSD-3-Clause license, and adds the
mask predictor and mask RoI head from the same ported commit.

</provenance-box>

## Citation

The authors publish no BibTeX block for this work. Cite the paper linked in
the Upstream row above.
