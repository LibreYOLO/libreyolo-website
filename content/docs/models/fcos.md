---
title: FCOS
families: [fcos]
seo_title: "FCOS in LibreYOLO: predict, validate and export"
description: "Run FCOS in LibreYOLO for anchor-free object detection. Install, predict, validate and export the BSD-3-Clause torchvision port, ResNet-50/FPN."
lead: "FCOS detects objects per pixel instead of relying on a set of predefined anchor boxes, predicting a box and a centerness score at every location on the feature map. LibreYOLO ports the torchvision implementation for detection."
keywords: [FCOS, anchor-free detection, object detection, one-stage detector, torchvision]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreFCOSr50.pt source=bus.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreFCOSr50.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

FCOS needs no optional extra. Everything it imports is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in
a different detector is a one line change. Calling the model with no
threshold arguments applies FCOS's own published defaults, `conf=0.2`,
`iou=0.6` and `max_det=100`; pass any of the three to override them. FCOS
keeps a final NMS step over its per-pixel predictions. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

One size: ResNet-50 with a feature pyramid, the only variant this family
recognizes.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained
on.

<code-tabs name="val" />

## Export

<export-matrix />

FCOS exports to ONNX, TorchScript and OpenVINO. FCOS preserves the source
aspect ratio before the graph runs, so LibreYOLO forces `dynamic=True` for the
ONNX and OpenVINO paths regardless of what is passed, to keep the graph valid
for padded input shapes. An exported `.onnx` file loads back through
`LibreYOLO()` on its file suffix and returns the same `Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
