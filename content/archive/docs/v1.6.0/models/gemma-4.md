---
title: Gemma 4
families:
  - gemma4
seo_title: Gemma 4 in LibreYOLO
description: Gemma 4 produces object detections from image and text input.
lead: Gemma 4 produces object detections from image and text input.
keywords:
  - Gemma 4
  - LibreYOLO
  - detect
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("gemma-4-e2b", device="cpu")
        model.set_classes(["person", "building"])
        result = model(SAMPLE_IMAGE)
        print(result.boxes)
---

## Install

```bash
pip install "libreyolo[vlm]" "transformers>=5.10.0"
```

## Predict

<code-tabs name="predict" />

The E2B and E4B adapters parse boxes on a 0–1000 coordinate grid. The bare `gemma-4` alias selects E4B. Install Transformers 5.10 or later. Training is not supported.

## Licensing

<provenance-box></provenance-box>
