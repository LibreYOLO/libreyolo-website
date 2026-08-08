---
title: YOLOv3
families: [yolo3]
seo_title: "YOLOv3 in LibreYOLO: predict, validate, export"
description: "Run YOLOv3 in LibreYOLO: a frozen, inference-only museum family with tiny, base and SPP sizes. Predict, validate and export, under a public-domain license."
lead: "YOLOv3 is the Darknet-53 detector that added multi-scale prediction and independent logistic classifiers to the YOLO line. LibreYOLO carries it as a frozen, inference-only exhibit in tiny, base and SPP sizes."
keywords: [YOLOv3, Darknet, Darknet-53, object detection, multi-scale detection, museum family]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO3b.pt source=bus.jpg save=True
    - label: SPP size
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The SPP variant adds a spatial pyramid pooling block before the
        # detection heads and runs at its own native input size.
        model = LibreYOLO("LibreYOLO3spp.pt")
        results = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreYOLO3b.onnx")
        results = model(SAMPLE_IMAGE)

        print(results[0].boxes.xyxy)
---

## Install

YOLOv3 needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

This family is inference-only: `train()` raises `NotImplementedError`, so this
page has no Train section. Predict, validate and export are all supported.
Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in
a different detector is a one line change. `conf` filters the confidence
threshold and `iou` the NMS threshold, applied per scale before boxes from all
three heads are merged. See [prediction](/docs/predict) for sources, streaming
and result handling.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you validate
on.

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
