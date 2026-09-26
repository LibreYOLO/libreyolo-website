---
title: DEKR
families:
  - dekr
seo_title: DEKR in LibreYOLO
description: >-
  DEKR estimates multi-person poses from image-wide keypoint heatmaps and
  offsets.
lead: >-
  DEKR estimates multi-person poses from image-wide keypoint heatmaps and
  offsets.
keywords:
  - DEKR
  - LibreYOLO
  - pose
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Downloads the W32 checkpoint from the upstream CDN and checks its SHA-256
        model = LibreYOLO("LibreDEKRw32-pose.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.keypoints)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Downloads the W32 checkpoint from the upstream CDN and checks its SHA-256
        model = LibreYOLO("LibreDEKRw32-pose.pt", device="cpu")
        # coco8-pose.yaml builds a 4-image COCO keypoint split on first use
        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True, workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Downloads the W32 checkpoint from the upstream CDN and checks its SHA-256
        model = LibreYOLO("LibreDEKRw32-pose.pt", device="cpu")
        model.export(format="onnx")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The adapter uses the W32 no-deformable-convolution graph and 17 COCO person keypoints. Boxes enclose confident decoded joints. This is an inference-only family; the original deformable graph is a different checkpoint architecture.


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Export

<export-matrix />

<code-tabs name="export" />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
