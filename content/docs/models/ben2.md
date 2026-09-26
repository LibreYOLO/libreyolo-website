---
title: BEN2
families:
  - ben2
seo_title: 'BEN2: prediction and training in LibreYOLO'
description: BEN2 predicts a soft foreground alpha matte for background removal.
lead: BEN2 predicts a soft foreground alpha matte for background removal.
keywords:
  - BEN2
  - LibreYOLO
  - matte
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBEN2b-matte.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        print(result.matte.array.shape)
        result.save("cutout.png")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

Prediction uses a fixed 1024-pixel canvas. Native prediction accepts batches; exports require batch 1. `result.save()` writes a transparent cutout. Training is not supported.

## Export

<export-matrix />

[Export setup](/docs/export) lists format dependencies and loading exported artifacts.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
