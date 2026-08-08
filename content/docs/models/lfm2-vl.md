---
title: LFM2-VL
families: [lfm2vl]
seo_title: "LFM2-VL: open-vocabulary detection in LibreYOLO"
description: "Use LFM2-VL in LibreYOLO for open-vocabulary object detection on-device. Predict with any text label; training, validation and export are not supported."
lead: "LFM2-VL is a compact, on-device vision-language model released by Liquid AI. LibreYOLO wraps it as an open-vocabulary object detector: any list of text labels becomes the class set, with no fixed head and no fine-tuning required."
keywords: [LFM2-VL, LFM2, Liquid AI, vision-language model, open-vocabulary detection, VLM, edge VLM, LibreVLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # Open vocabulary: any words work, not a fixed class head. Sticky
        # across every later predict()/track() call until set again.
        model.set_classes(["person", "bicycle", "dog"])
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Raw chat
      language: python
      code: |
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # The escape hatch beneath the detection convenience: free-form
        # questions, counting, or any prompt the boxes wrapper doesn't cover.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
---

## Install

LFM2-VL needs the `vlm` extra, which pulls in `transformers` for the
chat-template backbone.

```bash
pip install "libreyolo[vlm]"
```

## Predict

`LibreLFM2VL` is a Python class, not a `.pt` checkpoint: it is not loaded
through the `LibreYOLO()` factory, and the `libreyolo` CLI does not resolve
it. The `LibreVLM(...)` factory (`from libreyolo import LibreVLM`) also
reaches this family by alias, e.g. `LibreVLM("lfm2-vl-450m")`; the class used
below is what it constructs. Weights come from Liquid AI's own Hugging Face
repository, not a LibreYOLO mirror; the first call downloads and caches them
locally, and logs a one-time license notice before it does.

<code-tabs name="predict" />

`results[0].boxes` carries the parsed detections like any other family.
Confidence is a placeholder: LFM2-VL emits no per-box score, so every
detection gets the same constant confidence, and `conf=` only drops rows
below that constant, it does not rank them. `iou` discards near-duplicate
boxes of the same class above the given overlap, a side effect of greedy
decoding repeating an object; it is not a class-wise NMS pass. Skip
`set_classes()` and the vocabulary defaults to the COCO-80 names. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Two sizes: 450m and 1.6b, both from Liquid AI's LFM2.5-VL release, built for
on-device deployment. LibreYOLO's benchmark harness has not measured this
family, so there are no published accuracy numbers to compare them by; pick a
size against your own compute budget.

LibreYOLO exposes this family for prediction only. `train()`, `val()` and
`export()` all raise `NotImplementedError`: fine-tune upstream and load the
result instead, dataset validation is skipped because a placeholder
confidence would make COCO mAP misleading, and export is out of scope for a
generative model with no state dict to trace.

## Licensing

<provenance-box>

The LFM Open License v1.0 permits commercial use, reproduction and
modification, but only below a $10 million annual revenue threshold; a legal
entity at or above that threshold is not licensed under this agreement at all
for commercial use, and must contact Liquid AI directly. Qualified
non-profit organizations are exempt from the threshold for non-commercial or
research use. LibreYOLO ships no LiquidAI source code, since the model loads
through the Apache-2.0 `transformers` library, and does not host or
redistribute the weights: `LibreLFM2VL` downloads the matching size directly
from Liquid AI's own Hugging Face repository the first time it runs, and
logs a one-time notice before that download.

</provenance-box>

## Citation

<citation-block />
