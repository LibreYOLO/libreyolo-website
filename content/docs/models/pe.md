---
title: Perception Encoder
families:
  - pe
seo_title: 'Perception Encoder: prediction and training in LibreYOLO'
description: >-
  Perception Encoder maps images, text and finite video clips into a shared
  embedding space.
lead: >-
  Perception Encoder maps images, text and finite video clips into a shared
  embedding space.
keywords:
  - Perception Encoder
  - LibreYOLO
  - classify
  - embed
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePEt16-cls.pt", device="cpu")
        model.set_classes(["person", "building"])
        result = model(SAMPLE_IMAGE)
        print(result.probs)
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

Set classes before zero-shot classification. Select `task="embed"` when loading for embeddings. Video embedding samples `clip_frames=8` frames by default, averages their vectors and L2-normalizes the result. Training is not supported. Exported video graphs take direct runtime input.

## Export

<export-matrix />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
