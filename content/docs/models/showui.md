---
title: ShowUI
families:
  - showui
seo_title: ShowUI in LibreYOLO
description: ShowUI locates the target of an instruction as an image point.
lead: ShowUI locates the target of an instruction as an image point.
keywords:
  - ShowUI
  - LibreYOLO
  - point
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreGround, SAMPLE_IMAGE

        model = LibreGround("showui-2b", device="cpu")
        result = model(SAMPLE_IMAGE, prompt="the person")
        print(result.points.xy)
---

## Install

```bash
pip install "libreyolo[ground]"
```

## Predict

<code-tabs name="predict" />

Use `prompt=` or `query=` for one call, or `set_query()` for a persistent instruction. A list of instructions on one image returns at most one click per query. Multi-query calls on image lists or folders raise an error. Coordinates refer to the original image. Training, validation and export are not supported. See the [grounding API](/docs/reference/ground-api).

## Licensing

<provenance-box></provenance-box>
