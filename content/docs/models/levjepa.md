---
title: LeVJEPA
families:
  - levjepa
seo_title: 'LeVJEPA: prediction and training in LibreYOLO'
description: LeVJEPA produces clip embeddings and spatial patch tokens from video.
lead: LeVJEPA produces clip embeddings and spatial patch tokens from video.
keywords:
  - LeVJEPA
  - LibreYOLO
  - embed
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLeVJEPAl-embed.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.embeddings)
        print(model.embed_tokens(SAMPLE_IMAGE).shape)
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The encoder uses 16 frames at 224 pixels. Finite videos use a centered window sampled at approximately 7.5 FPS. `embed_tokens()` exposes patch tokens. Training is not supported. The TorchScript graph requires direct clip input with batch 1. Pretrained weights are CC-BY-NC-4.0.

## Export

<export-matrix />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
