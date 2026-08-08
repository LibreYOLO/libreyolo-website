---
title: RetinaNet
families: [retinanet]
seo_title: "RetinaNet in LibreYOLO: predict, validate and export"
description: "Run RetinaNet in LibreYOLO for one-stage object detection with focal loss. Install, predict, validate and export the BSD-3-Clause torchvision port."
lead: "RetinaNet is a one-stage detector trained with focal loss, which down-weights easy negatives so a dense grid of anchors no longer needs a separate proposal stage to stay accurate. LibreYOLO ports the torchvision implementation for detection."
keywords: [RetinaNet, focal loss, object detection, one-stage detector, torchvision]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRetinaNetr50v2.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreRetinaNetr50v2.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

RetinaNet needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in
a different detector is a one line change. `conf` and `iou` set the
confidence and NMS thresholds; RetinaNet keeps its upstream NMS step over the
dense anchor grid. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Variants

Two sizes, both ResNet-50 with a feature pyramid: `r50` is the original
head, and `r50v2` replaces it with a GroupNorm head and a wider P6 block fed
from the backbone's last stage instead of the FPN output.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained
on.

<code-tabs name="val" />

## Export

<export-matrix />

RetinaNet exports to ONNX only, at batch size 1. RetinaNet resizes to a
variable, aspect-preserved input, so LibreYOLO forces `dynamic=True`
regardless of what is passed, to keep the graph valid for sources of
different shapes. An exported `.onnx` file loads back through `LibreYOLO()`
on its file suffix and returns the same `Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
