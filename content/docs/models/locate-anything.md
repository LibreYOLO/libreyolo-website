---
title: LocateAnything
families: [locateanything]
seo_title: "LocateAnything: open-vocabulary detection and pointing"
description: "Use LocateAnything in LibreYOLO for open-vocabulary detection and pointing. Predict with any text label; training, validation and export are not supported."
lead: "LocateAnything is a vision-language grounding model released by NVIDIA that decodes bounding boxes and points in parallel rather than one coordinate token at a time. LibreYOLO wraps it as an open-vocabulary detector and pointer: any list of text labels becomes the class set, with no fixed head and no fine-tuning required."
keywords: [LocateAnything, NVIDIA, vision-language model, open-vocabulary detection, point detection, VLM, grounding, LibreVLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # Open vocabulary: any words work, not a fixed class head. Sticky
        # across every later predict()/track() call until set again.
        model.set_classes(["person", "bicycle", "dog"])
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Point prompting
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        # task="point" returns one point per matched object instead of a box.
        # Switch tasks on an already-loaded model with model.set_task("point").
        model = LibreLocateAnything(size="3b", task="point")
        model.set_classes(["the person closest to the camera"])
        result = model(SAMPLE_IMAGE, save=True)

        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Raw chat
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # The escape hatch beneath the detection convenience: free-form
        # questions, counting, or any prompt the boxes wrapper doesn't cover.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
---

## Install

LocateAnything needs the `vlm` extra, which pulls in `transformers` plus the
`decord`, `lmdb` and `peft` packages its Hugging Face remote code imports at
load time.

```bash
pip install "libreyolo[vlm]"
```

## Predict

`LibreLocateAnything` is a Python class, not a `.pt` checkpoint: it is not
loaded through the `LibreYOLO()` factory, and the `libreyolo` CLI does not
resolve it. The `LibreVLM(...)` factory (`from libreyolo import LibreVLM`)
also reaches this family by alias, e.g. `LibreVLM("locate-anything")`; the
class used below is what it constructs. Loading it downloads and executes
NVIDIA's own remote model code from Hugging Face, so LibreYOLO pins the
download to one fixed commit revision rather than the mutable `main` branch,
and logs a one-time license notice before the first download.

<code-tabs name="predict" />

`result.boxes` (task `detect`) and `result.points` (task `point`)
carry the parsed output like any other family. Confidence is a placeholder:
LocateAnything emits no per-box score, so every detection gets the same
constant confidence, and `conf=` only drops rows below that constant, it
does not rank them. Skip `set_classes()` and the vocabulary defaults to the
COCO-80 names. See [prediction](/docs/predict) for sources, streaming and
result handling.

## Variants

One published size, 3b. Two tasks share the same weights: `detect` (the
default) returns boxes, and `task="point"` returns a single point per
matched object instead, in `result.points`; switch between them on an
already-loaded model with `model.set_task("point")`. LibreYOLO's benchmark
harness has not measured this family, so there are no published accuracy
numbers to compare against.

LibreYOLO exposes this family for prediction only. `train()`, `val()` and
`export()` all raise `NotImplementedError`: fine-tune upstream and load the
result instead, dataset validation is skipped because a placeholder
confidence would make COCO mAP misleading, and export is out of scope for a
generative model with no state dict to trace.

## Licensing

<provenance-box>

The NVIDIA License permits use, reproduction and modification, but restricts
the model and any derivative to non-commercial use, research or evaluation
only, for anyone other than NVIDIA and its affiliates: there is no revenue
threshold or paid exception. LocateAnything-3B also composes two other
licensed components: a Qwen2.5-3B-Instruct language backbone under the Qwen
Research License, and a MoonViT-SO-400M vision encoder under MIT. LibreYOLO
does not host, mirror or redistribute any of it: `LibreLocateAnything`
downloads the weights and the required remote code directly from
`nvidia/LocateAnything-3B` on Hugging Face, pinned to one fixed commit, the
first time it runs.

</provenance-box>

## Citation

<citation-block />
