---
title: PP-LiteSeg
families:
  - ppliteseg
seo_title: PP-LiteSeg in LibreYOLO
description: >-
  PP-LiteSeg assigns a semantic class to each pixel using an STDC backbone and a
  fusion decoder.
lead: >-
  PP-LiteSeg assigns a semantic class to each pixel using an STDC backbone and a
  fusion decoder.
keywords:
  - PP-LiteSeg
  - LibreYOLO
  - semantic
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPLiteSegt50-sem.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.semantic_mask)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPLiteSegt50-sem.pt", device="cpu")
        # Semantic segmentation dataset YAML: images plus single-channel class-ID masks.
        model.train(data="path/to/your/semantic.yaml", epochs=1, device="cpu", workers=0)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPLiteSegt50-sem.pt", device="cpu")
        # Semantic segmentation dataset YAML with 19 Cityscapes classes, like cityscapes.yaml.
        metrics = model.val(data="path/to/your/semantic.yaml", workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPLiteSegt50-sem.pt", device="cpu")
        model.export(format="onnx")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The t50 and b50 variants evaluate at `(512, 1024)`; t75 and b75 evaluate at `(768, 1536)`. Dimensions are `(height, width)`.

## Train

Training defaults to 800 epochs, batch 8 and `amp=False`. The default crop is `(512, 1024)` for t50/b50 and `(768, 768)` for t75/b75. Validation keeps the native rectangle. The recipe combines cross-entropy, Dice and edge losses with auxiliary heads.

[Dataset setup](/docs/train/datasets) describes the required training data.

<code-tabs name="train" />


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Export

<export-matrix />

<code-tabs name="export" />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
