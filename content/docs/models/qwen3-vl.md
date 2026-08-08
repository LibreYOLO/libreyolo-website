---
title: Qwen3-VL
families: [qwen3vl]
seo_title: "Qwen3-VL in LibreYOLO: open-vocabulary detection"
description: "Qwen3-VL in LibreYOLO: install, set an open vocabulary and predict or chat with Alibaba's Apache-2.0 vision-language model."
lead: "Qwen3-VL is Alibaba's vision-language model with native 2D grounding. LibreYOLO wraps it as an open-vocabulary object detector and exposes its free-form chat directly: supply a class list to detect, or ask it a question."
keywords: [Qwen3-VL, vision-language model, open-vocabulary detection, grounding, Alibaba, VLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        results = model.predict(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")

        # The escape hatch beneath the detection convenience: any question,
        # not just a bounding-box query.
        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety vest?")
        print(answer)
---

## Install

Qwen3-VL belongs to LibreYOLO's VLM-as-detector tier, a separate product
surface from the checkpoint-based families with its own factory. It needs the
`vlm` extra.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.
`LibreVLM()` called with no argument defaults to Qwen3-VL-4B.

<code-tabs name="predict" />

This family loads through the `LibreVLM()` factory, not `LibreYOLO()`: VLM
families declare no checkpoint loader, so the file-suffix routing described on
other model pages does not apply here. `set_classes()` sets the vocabulary
Qwen3-VL is asked to find; it is sticky, so it stays in effect across every
later `predict()`/`track()` call until you set it again. Every detection
carries the same placeholder confidence, so `conf` filtering is all-or-nothing
rather than a ranking; `iou` does have an effect for this family, dropping a
later same-class box once it overlaps an already-kept one past the threshold,
since a repeating generator can otherwise emit near-duplicate boxes for one
object. Unlike Florence-2 and Kosmos-2, Qwen3-VL also answers free-form
questions through `chat()`, the same escape hatch documented on the `LibreVLM`
factory. LibreYOLO's CLI does not cover this tier: there is no
`libreyolo predict model=...` form for it. See [prediction](/docs/predict) for
sources, streaming and result handling.

## Variants

Three sizes: Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct and Qwen3-VL-8B-Instruct,
loaded as `LibreVLM("qwen3-vl-2b")`, `LibreVLM("qwen3-vl-4b")` and
`LibreVLM("qwen3-vl-8b")`. All three declare a nominal 1024 px input, but the
Qwen processor's own smart-resize decides the actual canvas passed to the
network, so that figure is not a fixed operating resolution the way it is for
the other families on this site. LibreYOLO has not published a benchmark
comparing accuracy across the three sizes.

LibreYOLO does not train, validate or export Qwen3-VL: `train()`, `val()` and
`export()` all raise `NotImplementedError` for every family in this tier (see
the support tier above). Fine-tune Qwen3-VL upstream and load the resulting
weights if you need a custom vocabulary baked in; check `predict()` output by
eye instead of a COCO-style validation pass, since every detection carries the
same placeholder confidence.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
