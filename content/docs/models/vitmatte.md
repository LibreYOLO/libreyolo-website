---
title: ViTMatte
families:
  - vitmatte
seo_title: ViTMatte in LibreYOLO
description: ViTMatte predicts a foreground alpha matte from an image and a trimap.
lead: ViTMatte predicts a foreground alpha matte from an image and a trimap.
keywords:
  - ViTMatte
  - LibreYOLO
  - matte
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreViTMattes-matte.pt", device="cpu")

        import numpy as np

        from PIL import Image


        image = Image.open(SAMPLE_IMAGE).convert("RGB")

        # An all-unknown guide demonstrates the input contract. Supply a real
        trimap for matting.

        trimap = np.full((image.height, image.width), 128, dtype=np.uint8)

        result = model(image, trimap=trimap)

        result.save("cutout.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTMattes-matte.pt", device="cpu")
        # A folder with images/ and mattes/ (grayscale alpha mattes, same file stems), or a matte YAML.
        metrics = model.val(data="path/to/your/matte_dataset", workers=0)
        print(metrics)
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

A trimap marks background as 0, unknown pixels as 128 and foreground as 255. Pass it through `trimap=` for a single image. Validation accepts `trimap_dir` or derives guides with `trimap_radius`. Training and export are not supported.


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
