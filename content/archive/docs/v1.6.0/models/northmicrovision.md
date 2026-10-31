---
title: North Micro Vision
families:
  - northmicrovision
seo_title: North Micro Vision in LibreYOLO
description: North Micro Vision detects objects through a text vocabulary.
lead: North Micro Vision detects objects through a text vocabulary.
keywords:
  - North Micro Vision
  - LibreYOLO
  - detect
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("north-micro-vision", device="cpu")
        model.set_classes(["person", "building"])
        result = model(SAMPLE_IMAGE)
        print(result.boxes)
---

## Install

```bash
pip install "libreyolo[vlm]" "transformers>=5.16.0"
```

## Predict

<code-tabs name="predict" />

The adapter runs one detection query per vocabulary class. It requires Transformers 5.16 or later, above the shared VLM extra floor. Training is not supported.

## Licensing

<provenance-box></provenance-box>
