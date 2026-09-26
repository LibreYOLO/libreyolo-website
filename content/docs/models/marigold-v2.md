---
title: Marigold V2
families:
  - marigold_v2
seo_title: Marigold V2 in LibreYOLO
description: >-
  Marigold V2 estimates depth, surface normals or intrinsic albedo with
  task-specific diffusion adapters.
lead: >-
  Marigold V2 estimates depth, surface normals or intrinsic albedo with
  task-specific diffusion adapters.
keywords:
  - Marigold V2
  - LibreYOLO
  - depth
  - normal
  - albedo
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Requires CUDA for default four-bit inference.
        model = LibreYOLO("LibreMarigoldV2b-depth.pt", device="cuda")
        result = model(SAMPLE_IMAGE)
        print(result)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Requires CUDA for default four-bit inference.
        model = LibreYOLO("LibreMarigoldV2b-depth.pt", device="cuda")
        metrics = model.val(data=input("Validation dataset path: "), workers=0)
        print(metrics)
---

## Install

```bash
pip install "libreyolo[marigold]"
```

The default four-bit runtime requires CUDA.

## Predict

<code-tabs name="predict" />

The adapter downloads a separately pinned base model. Default `quantization="4bit"` requires CUDA; the snippet requires that runtime. `quantization="none"` selects the CPU-capable path. The default seed is 2025. Albedo results hold linear RGB and convert to sRGB for display. Training and export are not supported.


## Validate

<code-tabs name="val" />

Use the dataset format for this task. [Validation](/docs/train/validation) explains the dataset requirements and returned metrics.

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>
