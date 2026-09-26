---
title: Molmo2
families:
  - molmo2
seo_title: Molmo2 in LibreYOLO
description: Molmo2 locates objects as points using a text vocabulary.
lead: Molmo2 locates objects as points using a text vocabulary.
keywords:
  - Molmo2
  - LibreYOLO
  - point
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("molmo2-4b", device="cpu")
        model.set_classes(["person", "building"])
        result = model(SAMPLE_IMAGE)
        print(result.points)
---

## Install

```bash
pip install "libreyolo[molmo2]"
```

## Predict

<code-tabs name="predict" />

`set_classes()` defines the pointing vocabulary. A custom pointing template must contain `{label}`. The factory also recognizes 8B and O-7B aliases; those snapshots were not verified by the release audit. Training is not supported.

## Licensing

<provenance-box></provenance-box>
