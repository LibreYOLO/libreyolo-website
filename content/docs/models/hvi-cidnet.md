---
title: HVI-CIDNet
families:
  - hvi_cidnet
seo_title: 'HVI-CIDNet: prediction and training in LibreYOLO'
description: >-
  HVI-CIDNet restores low-light images through hue, saturation and intensity
  processing.
lead: >-
  HVI-CIDNet restores low-light images through hue, saturation and intensity
  processing.
keywords:
  - HVI-CIDNet
  - LibreYOLO
  - restore
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHVICIDNett-restore.pt", device="cpu")
        result = model(SAMPLE_IMAGE, gamma=1.0, saturation=1.0, intensity=1.0)
        result.save("enhanced.png")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

`gamma`, `saturation` and `intensity` each default to 1.0. Validation uses paired image data. Training and export are not supported.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
