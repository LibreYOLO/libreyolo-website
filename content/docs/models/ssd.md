---
title: SSD
families: [ssd]
seo_title: "SSD (SSD300): object detection in LibreYOLO"
description: "Run SSD300 in LibreYOLO: a single-shot VGG16 detector for prediction, validation and ONNX export under BSD-3-Clause. No training path."
lead: "SSD (Single Shot MultiBox Detector) predicts every box and class score from a dense grid of default boxes in one forward pass, with no separate region-proposal stage. LibreYOLO ships the VGG16-backed SSD300 checkpoint as an inference-only detector."
keywords: [SSD, SSD300, Single Shot MultiBox Detector, object detection, VGG16, anchor-based detector]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSSD300.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")

        # imgsz is left out here on purpose: SSD300 traces at its checkpoint's
        # native canvas, and any other value raises before export starts.
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreSSD300.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

SSD needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. SSD decodes its default-box grid with
per-class scores and then runs non-maximum suppression, so `conf`, `iou` and
`max_det` all have a real effect here, unlike the query-based detectors in this
library. See [prediction](/docs/predict) for sources, streaming and result
handling.

## Variants

SSD ships one checkpoint: the VGG16-backed SSD300 network at its fixed native
canvas. There is no size or scale choice in this family; predict, validate and
export all use that one graph.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

SSD exports to ONNX only; every other format is currently blocked for this
family. Export always uses the checkpoint's native canvas, and the graph
exposes SSD's raw packed head rather than a fused non-maximum-suppression
output, so `nms=True` is not accepted at export time. LibreYOLO's own backends
run the decode and suppression step after loading the graph back.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

LibreYOLO's SSD300 code is not ported from the paper authors' own Caffe
release; it derives from torchvision's BSD-3-Clause SSD300 implementation, and
that is the repository linked above as the upstream source. The backbone's
VGG16 weights trace further back to Oxford's fully convolutional reduced
VGGNet, released under CC BY 4.0 by Karen Simonyan and Andrew Zisserman.

</provenance-box>

## Citation

<citation-block />
