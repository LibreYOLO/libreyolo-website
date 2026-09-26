---
title: HVI-CIDNet
families:
  - hvi_cidnet
seo_title: HVI-CIDNet in LibreYOLO
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
        result.restored.save("enhanced.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHVICIDNett-restore.pt", device="cpu")
        # Paired restoration dataset YAML: degraded images in inputs/, clean targets in targets/, matched by file stem.
        metrics = model.val(data="path/to/your/restore.yaml", workers=0)
        print(metrics)
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

`gamma`, `saturation` and `intensity` each default to 1.0. Validation uses paired image data. Training and export are not supported.


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
