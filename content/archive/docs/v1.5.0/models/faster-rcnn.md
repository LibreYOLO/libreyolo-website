---
title: Faster R-CNN
families: [faster_rcnn]
seo_title: "Faster R-CNN in LibreYOLO: predict, validate and export"
description: "Run Faster R-CNN in LibreYOLO for object detection across four backbones. Install, predict, validate and export the BSD-3-Clause torchvision port."
lead: "Faster R-CNN detects objects with a region proposal network feeding a two-stage classifier, the architecture that made region proposals part of the same trained network instead of a separate step. LibreYOLO ports the torchvision implementation for detection."
keywords: [Faster R-CNN, object detection, region proposal network, two-stage detector, torchvision]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFasterRCNNl.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreFasterRCNNl.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

Faster R-CNN needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in
a different detector is a one line change. `conf` and `iou` set the
confidence and NMS thresholds; Faster R-CNN keeps its upstream NMS step,
unlike a query-based detector. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Variants

Four sizes, each a different torchvision configuration rather than a scaled
version of the same one: `n` is MobileNetV3-Large at a 320 px input, `s` is
the same backbone at 800 px, `m` is ResNet-50 with a feature pyramid, and `l`
is the v2 revision, with a deeper region proposal head and a four-convolution
box head in place of `m`'s. `n` and `s` trade accuracy for a lighter
backbone.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained
on.

<code-tabs name="val" />

## Export

<export-matrix />

Faster R-CNN exports to ONNX only, at batch size 1. The exported graph keeps
the upstream resize step inside it, so LibreYOLO forces `dynamic=True`
regardless of what is passed, to keep the graph valid for sources that are
not square. An exported `.onnx` file loads back through `LibreYOLO()` on its
file suffix and returns the same `Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
