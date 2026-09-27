---
title: LaMa
families:
  - lama
seo_title: LaMa in LibreYOLO
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

        result.restored.save("inpainted.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLaMab-restore.pt", device="cpu")
        # Restore dataset YAML with paired inputs/targets folders plus a mask_dir key
        metrics = model.val(data="path/to/your/restore.yaml", workers=0)
        print(metrics)
---

## Install

```bash
pip install "libreyolo[onnx]"
```

## Predict

<code-tabs name="predict" />

Pass `mask=` for single-image inference: nonzero pixels mark the region to fill. The checkpoint embeds an ONNX graph and requires ONNX Runtime 1.18 or later. Training and export are not supported.


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
