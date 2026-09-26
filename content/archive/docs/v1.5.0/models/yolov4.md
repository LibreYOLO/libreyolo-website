---
title: YOLOv4
families: [yolo4]
seo_title: "YOLOv4: run, validate and export in LibreYOLO"
description: "Run YOLOv4 in LibreYOLO: a frozen, inference-only museum family with a CSPDarknet-53 backbone. Predict, validate and export, under a public-domain license."
lead: "YOLOv4 combines a CSPDarknet-53 backbone, an SPP block and a PANet neck with Mish activations. LibreYOLO carries it as a frozen, inference-only exhibit in tiny and base sizes."
keywords: [YOLOv4, Darknet, CSPDarknet-53, PANet, object detection, Mish activation, museum family]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO4b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO4b.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO4b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO4b.pt format=onnx
        libreyolo export model=LibreYOLO4b.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreYOLO4b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

YOLOv4 needs no extra beyond the base package.

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
threshold and `iou` the NMS threshold, applied after each head's own
`scale_x_y` center scaling. See [prediction](/docs/predict) for sources,
streaming and result handling.

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

## Citation

<citation-block />
