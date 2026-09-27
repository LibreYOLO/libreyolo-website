---
title: LibreGround API
seo_title: LibreGround API | LibreYOLO
description: >-
  LibreGround maps an image and a referring instruction to at most one point per
  query.
lead: >-
  LibreGround maps an image and a referring instruction to at most one point per
  query.
keywords:
  - LibreGround API
  - LibreYOLO
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGround, SAMPLE_IMAGE

        model = LibreGround(device="cpu")
        result = model(SAMPLE_IMAGE, prompt="the person")
        print(result.points.xy)
---

## Load

Install `libreyolo[ground]`. `LibreGround()` selects ShowUI-2B; Florence-2 base/large and Qwen3-VL 2B/4B/8B adapters are also available.

## Predict

<code-tabs name="predict" />

`prompt=` and `query=` set an instruction for that call. `set_query()` persists it. A single image can receive a query list; each query contributes at most one point and its query index becomes the class ID. Query lists with folders or image lists raise.

`Results.points.xy` contains original-image pixel coordinates. Confidence is a placeholder of 1.0. The factory provides inference only; it has no training, validation or export path.
