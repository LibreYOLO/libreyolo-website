---
title: CenterNet
families: [centernet]
seo_title: "CenterNet: object detection in LibreYOLO"
description: "Run CenterNet (Objects as Points) in LibreYOLO with the ResDCN-18 and DLA-34 backbones. Predict, validate and export to ONNX under MIT. No training path."
lead: "CenterNet models an object as the center point of its bounding box and regresses every other property from a heatmap peak, so it needs no anchors and no non-maximum-suppression step. LibreYOLO ships it as an inference-only detector."
keywords: [CenterNet, Objects as Points, keypoint detection, anchor-free detector, ResDCN-18, DLA-34]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreCenterNetresdcn18.pt source=bus.jpg save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")

        # ONNX export needs opset 16 or newer: the deformable-convolution
        # upsampling stage lowers to GridSample, which opset 16 introduced.
        model.export(format="onnx", opset=18)
        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreCenterNetresdcn18.onnx")
        result = model("bus.jpg")

        print(result.boxes.xyxy)
---

## Install

CenterNet needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. `conf` and `max_det` filter the ranked
heatmap peaks; `iou` is accepted for API parity but has no effect, because
CenterNet's top-k peak decode needs no box-IoU suppression step. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Two backbones. `resdcn18` pairs a ResNet-18 trunk with deformable-convolution
upsampling; `dla34` pairs a DLA-34 trunk with iterative deep-aggregation
upsampling. Both feed the same three dense heads (heatmap, width/height,
offset) and the same input canvas.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

ONNX export requires opset 16 or newer: the deformable-convolution upsampling
stage in both backbones lowers to the ONNX `GridSample` operator, which opset
16 introduced. Requesting an older opset raises before tracing starts.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

The ResDCN-18 graph also credits Microsoft's MIT-licensed
human-pose-estimation.pytorch, and the DLA-34 graph credits Fisher Yu's
BSD-3-Clause DLA implementation. LibreYOLO does not vendor the original DCNv2
extension the upstream project used; native execution runs torchvision's
BSD-3-Clause `deform_conv2d` instead, and the export-only portable
implementation was authored separately for LibreYOLO.

</provenance-box>

## Citation

<citation-block />
