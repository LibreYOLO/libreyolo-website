---
title: DEKR
families:
  - dekr
seo_title: 'DEKR: prediction and training in LibreYOLO'
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
      code: >
        from libreyolo import LibreDEKR, SAMPLE_IMAGE


        model = LibreDEKR(model_path=input("Path to upstream weights: "),
        device="cpu")

        result = model(SAMPLE_IMAGE)

        print(result.keypoints)
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The adapter uses the W32 no-deformable-convolution graph and 17 COCO person keypoints. Boxes enclose confident decoded joints. This is an inference-only family; the original deformable graph is a different checkpoint architecture.

## Export

<export-matrix />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
