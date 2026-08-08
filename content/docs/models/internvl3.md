---
title: InternVL3
families: [internvl3]
seo_title: "InternVL3: open-vocabulary detection in LibreYOLO"
description: "Use InternVL3 in LibreYOLO for open-vocabulary object detection. Predict with any text label; training, validation and export are not supported."
lead: "InternVL3 is a native multimodal large language model released by OpenGVLab that jointly learns vision and language in a single pre-training stage. LibreYOLO wraps it as an open-vocabulary object detector: any list of text labels becomes the class set, with no fixed head and no fine-tuning required."
keywords: [InternVL3, InternVL, vision-language model, open-vocabulary detection, VLM, OpenGVLab, LibreVLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # Open vocabulary: any words work, not a fixed class head. Sticky
        # across every later predict()/track() call until set again.
        model.set_classes(["person", "bicycle", "dog"])
        results = model(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Raw chat
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # The escape hatch beneath the detection convenience: free-form
        # questions, counting, or any prompt the boxes wrapper doesn't cover.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
---

## Install

InternVL3 needs the `vlm` extra, which pulls in `transformers` for the
chat-template backbone.

```bash
pip install "libreyolo[vlm]"
```

## Predict

`LibreInternVL3` is a Python class, not a `.pt` checkpoint: it is not loaded
through the `LibreYOLO()` factory, and the `libreyolo` CLI does not resolve it.
The `LibreVLM(...)` factory (`from libreyolo import LibreVLM`) also reaches
this family by alias, e.g. `LibreVLM("internvl3-2b")`; the class used below is
what it constructs. Weights come from OpenGVLab's own `-hf` Hugging Face
repositories, not a LibreYOLO mirror; the first call downloads and caches
them locally, and logs a one-time license notice for the gated Qwen weights
before it does.

<code-tabs name="predict" />

`results[0].boxes` carries the parsed detections like any other family.
Confidence is a placeholder: InternVL3 emits no per-box score, so every
detection gets the same constant confidence, and `conf=` only drops rows
below that constant, it does not rank them. `iou` discards near-duplicate
boxes of the same class above the given overlap, a side effect of greedy
decoding repeating an object; it is not a class-wise NMS pass. Skip
`set_classes()` and the vocabulary defaults to the COCO-80 names. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Three sizes: 1b, 2b and 8b, all OpenGVLab's native `-hf` checkpoints (a Qwen
LLM backbone, not the two-tower architecture the original InternVL paper
describes). LibreYOLO's benchmark harness has not measured this family, so
there are no published accuracy numbers to compare them by; pick a size
against your own compute budget.

LibreYOLO exposes this family for prediction only. `train()`, `val()` and
`export()` all raise `NotImplementedError`: fine-tune upstream and load the
result instead, dataset validation is skipped because a placeholder
confidence would make COCO mAP misleading, and export is out of scope for a
generative model with no state dict to trace.

## Licensing

<provenance-box>

InternVL3's own code is MIT, permissive and usable in commercial and
closed-source products. The `-hf` checkpoints this family loads carry a Qwen
LLM backbone and are licensed separately, under Alibaba Cloud's Qwen License:
free to use, modify and redistribute with a "Built with Qwen" or "Improved
using Qwen" attribution requirement, and a 100 million monthly-active-user
ceiling on commercial use above which Alibaba's own authorization is
required. LibreYOLO does not host or redistribute these weights:
`LibreInternVL3` downloads the matching size directly from
`OpenGVLab/InternVL3-<size>-hf` on Hugging Face the first time it runs, and
logs a one-time notice for the Qwen License before that download.

</provenance-box>

## Citation

<citation-block />
