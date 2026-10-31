---
title: DDColor
families:
  - ddcolor
seo_title: DDColor in LibreYOLO
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
        result.restored.save("colorized.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDDColort-restore.pt", device="cpu")
        # Restore dataset YAML pairing grayscale inputs/val with color targets/val by file stem
        metrics = model.val(data="path/to/your/restore.yaml", workers=0)
        print(metrics)
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

The tiny and large variants return restored color images. Validation uses paired image data. Training and export are not supported.


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
