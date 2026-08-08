---
title: Florence-2
families: [florence2]
seo_title: "Florence-2 in LibreYOLO: open-vocabulary detection"
description: "Florence-2 in LibreYOLO: install, set an open vocabulary and predict boxes with Microsoft's MIT-licensed vision model."
lead: "Florence-2 is Microsoft's vision foundation model, prompted with a task token instead of run through a fixed detection head. LibreYOLO wraps it as an open-vocabulary object detector: supply the class list at predict time."
keywords: [Florence-2, vision-language model, open-vocabulary detection, grounding, Microsoft, VLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        results = model.predict(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])

        # Any source the library accepts: file, folder, URL, webcam index,
        # RTSP stream, or a .streams list
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
---

## Install

Florence-2 belongs to LibreYOLO's VLM-as-detector tier, a separate product
surface from the checkpoint-based families with its own factory. It needs the
`vlm` extra.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.
LibreYOLO downloads the florence-community re-upload of the checkpoint rather
than the original `microsoft/Florence-2-*` repository; see Licensing for why.

<code-tabs name="predict" />

This family loads through the `LibreVLM()` factory, not `LibreYOLO()`: VLM
families declare no checkpoint loader, so the file-suffix routing described on
other model pages does not apply here. `set_classes()` sets the vocabulary
Florence-2 is asked to find in the image; it is sticky, so it stays in effect
across every later `predict()`/`track()` call until you set it again. The
returned `Results` carries `boxes` in the same shape as any other family, but
every detection carries the same placeholder confidence, so `conf` filtering is
all-or-nothing rather than a ranking, and `iou` has no effect: Florence-2's
wrapper builds the detection list directly from the parsed task-token output,
with no deduplication step. `chat()` raises `NotImplementedError` here, because
Florence-2 is driven by the `<OPEN_VOCABULARY_DETECTION>` task token rather than
a chat template. LibreYOLO's CLI does not cover this tier: there is no
`libreyolo predict model=...` form for it. See [prediction](/docs/predict) for
sources, streaming and result handling.

## Variants

Two sizes: Florence-2-base and Florence-2-large, both at 768 px, loaded as
`LibreVLM("florence-2-base")` or `LibreVLM("florence-2-large")`. LibreYOLO has
not published a benchmark comparing accuracy between them.

LibreYOLO does not train, validate or export Florence-2: `train()`, `val()` and
`export()` all raise `NotImplementedError` for every family in this tier (see
the support tier above). Fine-tune Florence-2 upstream and load the resulting
weights if you need a custom vocabulary baked in; check `predict()` output by
eye instead of a COCO-style validation pass, since every detection carries the
same placeholder confidence.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
