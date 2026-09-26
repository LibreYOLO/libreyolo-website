---
title: DDColor
families:
  - ddcolor
seo_title: 'DDColor: prediction and training in LibreYOLO'
description: DDColor predicts image color from luminance.
lead: DDColor predicts image color from luminance.
keywords:
  - DDColor
  - LibreYOLO
  - restore
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDDColort-restore.pt", device="cpu")
        result = model(SAMPLE_IMAGE)
        result.save("colorized.png")
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The tiny and large variants return restored color images. Validation uses paired image data. Training and export are not supported.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
