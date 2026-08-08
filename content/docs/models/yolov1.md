---
title: YOLOv1
families: [yolo1]
seo_title: "YOLOv1 in LibreYOLO: predict, validate, export"
description: "Run the original YOLOv1 detector in LibreYOLO: a frozen, inference-only museum family. Predict, validate and export, under a public-domain license."
lead: "YOLOv1 is the original 2016 detector that gave the YOLO family its name: one convolutional network with a fully connected head predicts every box and class score in a single pass, with no anchor boxes. LibreYOLO carries it as a frozen, inference-only exhibit."
keywords: [YOLOv1, YOLO v1, Darknet, object detection, Pascal VOC, museum family]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO1b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreYOLO1b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

YOLOv1 needs no extra beyond the base package.

```bash
pip install libreyolo
```

## Predict

This family is inference-only: `train()` raises `NotImplementedError`, so this
page has no Train section. Predict, validate and export are all supported.
Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in
a different detector is a one line change. Two things are specific to this
family. The published checkpoint is trained on Pascal VOC (2007+2012), not
COCO, so `box.cls` indexes the 20 VOC categories (aeroplane, bicycle, bird,
boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike,
person, pottedplant, sheep, sofa, train, tvmonitor) rather than the 80 COCO
ones. And the fully connected detection head accepts one image at a time, so a
list of sources is looped rather than run as a true batch. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against a dataset in the same VOC-style label
space the checkpoint was trained on.

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

## Citation

The authors publish no BibTeX block for this work. Cite the paper linked in
the Upstream row above.
