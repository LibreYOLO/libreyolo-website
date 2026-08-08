---
title: EfficientDet
families: [efficientdet]
seo_title: "EfficientDet: object detection in LibreYOLO"
description: "Run EfficientDet D0-D4 in LibreYOLO: BiFPN detectors for prediction, validation and export to ONNX, TensorRT and OpenVINO under Apache-2.0."
lead: "EfficientDet pairs an EfficientNet backbone with a repeated bi-directional feature pyramid network (BiFPN) and scales depth, width and resolution together across five sizes. LibreYOLO ships it as an inference-only detector."
keywords: [EfficientDet, BiFPN, EfficientNet, object detection, compound scaling]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEfficientDetd0.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Use the exported file
      language: python
      code: |
        from libreyolo import LibreYOLO

        # The factory routes on the file suffix, so an exported artifact loads
        # like any checkpoint and returns the same Results object.
        model = LibreYOLO("LibreEfficientDetd0.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## Install

EfficientDet needs no optional extra. Everything it imports is in the base
install.

```bash
pip install libreyolo
```

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

The returned `Results` object is the one every family returns, so swapping in a
different detector is a one line change. EfficientDet decodes anchor-based
candidates and then runs class-wise non-maximum suppression, so `conf`, `iou`
and `max_det` all have a real effect here. See [prediction](/docs/predict) for
sources, streaming and result handling.

## Variants

Five sizes, D0 through D4. Each step up pairs a larger EfficientNet backbone
with a deeper, wider BiFPN and a deeper prediction head, so parameter count and
compute grow together, following the paper's compound-scaling rule.

## Validate

`val()` returns a dictionary of `metrics/` keys covering precision, recall,
mAP 50 and mAP 50-95, measured against any dataset in the format you trained on.

<code-tabs name="val" />

## Export

<export-matrix />

An exported artifact loads back through `LibreYOLO()` on its file suffix, so a
`.onnx` or `.engine` file behaves like a checkpoint and returns the same
`Results`.

<code-tabs name="export" />

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

LibreYOLO's D0-D4 checkpoints are converted through the Apache-2.0
rwightman/efficientdet-pytorch project, which itself mirrors the official
TensorFlow-trained weights from google/automl without changing learned
tensors. No source from the LGPL-licensed
zylo117/Yet-Another-EfficientDet-Pytorch project was consulted or used.

</provenance-box>

## Citation

The authors publish no BibTeX block for this work. Cite the paper linked in
the Upstream row above.
