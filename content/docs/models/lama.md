---
title: LaMa
families:
  - lama
seo_title: 'LaMa: prediction and training in LibreYOLO'
description: LaMa fills masked regions of an image.
lead: LaMa fills masked regions of an image.
keywords:
  - LaMa
  - LibreYOLO
  - restore
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreLaMab-restore.pt", device="cpu")

        import numpy as np

        from PIL import Image


        image = Image.open(SAMPLE_IMAGE).convert("RGB")

        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        mask[image.height//3:2*image.height//3, image.width//3:2*image.width//3]
        = 255

        result = model(image, mask=mask)

        result.save("inpainted.png")
---

## Install

```bash
pip install "libreyolo[onnx]"
```

## Predict

<code-tabs name="predict" />

Pass `mask=` for single-image inference: nonzero pixels mark the region to fill. The checkpoint embeds an ONNX graph and requires ONNX Runtime 1.18 or later. Training and export are not supported.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
