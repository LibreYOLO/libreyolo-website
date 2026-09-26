---
title: ViTMatte
families:
  - vitmatte
seo_title: 'ViTMatte: prediction and training in LibreYOLO'
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
---

## Install

```bash
pip install "libreyolo"
```

## Predict

<code-tabs name="predict" />

A trimap marks background as 0, unknown pixels as 128 and foreground as 255. Pass it through `trimap=` for a single image. Validation accepts `trimap_dir` or derives guides with `trimap_radius`. Training and export are not supported.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
