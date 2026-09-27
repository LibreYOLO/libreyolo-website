---
title: V-JEPA 2
families:
  - vjepa2
seo_title: V-JEPA 2 in LibreYOLO
description: >-
  V-JEPA 2 produces video embeddings and supports training an attentive
  classification probe.
lead: >-
  V-JEPA 2 produces video embeddings and supports training an attentive
  classification probe.
keywords:
  - V-JEPA 2
  - LibreYOLO
  - embed
  - classify
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVJEPA2l256-embed.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.embeddings)
        tokens = model.embed_tokens(SAMPLE_IMAGE)
        print(tokens.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVJEPA2l256-cls-ssv2.pt", device="cpu")
        # Replace with your video dataset YAML (train/val manifests of '<video-path> <class-id>' lines).
        model.train(data="path/to/your/video_dataset.yaml", epochs=1, device="cpu", workers=0)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVJEPA2l256-cls-ssv2.pt", device="cpu")
        # Replace with your video dataset YAML; val() reads its val manifest of '<video-path> <class-id>' lines.
        metrics = model.val(data="path/to/your/video_dataset.yaml", workers=0)
        print(metrics)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVJEPA2l256-embed.pt", device="cpu")
        model.export(format="onnx")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

A still image is promoted to a clip. `embed_tokens()` returns `(B, T, H, W, D)` tokens; the public embedding normalizes their mean. Exported clip graphs require direct five-dimensional runtime input.

## Train

Only the attentive probe and linear classifier train; the encoder stays frozen. Use the classification task and a video-manifest dataset. Defaults are 10 epochs, batch 2, learning rate 0.001 and workers 0. Full encoder fine-tuning and pretraining are not supported.

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
