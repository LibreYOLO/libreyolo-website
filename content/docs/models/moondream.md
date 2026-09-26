---
title: Moondream
families:
  - moondream
seo_title: Moondream in LibreYOLO
description: 'Moondream detects objects, returns points and answers image questions.'
lead: 'Moondream detects objects, returns points and answers image questions.'
keywords:
  - Moondream
  - LibreYOLO
  - detect
  - point
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("moondream-2", device="cpu")
        model.set_classes(["person", "building"])
        result = model(SAMPLE_IMAGE)
        print(result.boxes)
---

## Install

```bash
pip install "libreyolo[vlm]"
```

## Predict

<code-tabs name="predict" />

The default selects Moondream 2; `moondream-3` selects the BSL 1.1 model. Use `task="point"` for points and `chat()` for native text responses. Training is not supported.

## Licensing

<provenance-box></provenance-box>
