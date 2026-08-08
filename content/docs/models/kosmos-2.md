---
title: Kosmos-2
families: [kosmos2]
seo_title: "Kosmos-2 in LibreYOLO: grounded object detection"
description: "Kosmos-2 in LibreYOLO: install, set an open vocabulary and predict grounded boxes with Microsoft's MIT-licensed model."
lead: "Kosmos-2 is Microsoft's grounding model: it captions an image, then locates each noun phrase in that caption with a box. LibreYOLO wraps it as an open-vocabulary object detector: supply the class list at predict time."
keywords: [Kosmos-2, vision-language model, grounding, open-vocabulary detection, Microsoft, VLM]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        results = model.predict(SAMPLE_IMAGE, save=True)

        for box in results[0].boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])

        # Any source the library accepts: file, folder, URL, webcam index,
        # RTSP stream, or a .streams list
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
---

## Install

Kosmos-2 belongs to LibreYOLO's VLM-as-detector tier, a separate product
surface from the checkpoint-based families with its own factory. It needs the
`vlm` extra.

```bash
pip install "libreyolo[vlm]"
```

## Predict

Weights download from Hugging Face on first use and are cached locally.
LibreYOLO loads Microsoft's own `microsoft/kosmos-2-patch14-224` repository
directly; unlike Florence-2, no community re-upload is needed here.

<code-tabs name="predict" />

This family loads through the `LibreVLM()` factory, not `LibreYOLO()`: VLM
families declare no checkpoint loader, so the file-suffix routing described on
other model pages does not apply here. `set_classes()` sets the vocabulary
Kosmos-2 is asked to find; it is sticky, so it stays in effect across every
later `predict()`/`track()` call until you set it again. Kosmos-2 grounds noun
phrases rather than matching a label exactly, so LibreYOLO's wrapper accepts a
partial match: a class named `"boat"` also matches a generated phrase like "the
boats". Every detection carries the same placeholder confidence, so `conf`
filtering is all-or-nothing rather than a ranking, and `iou` has no effect here,
since the wrapper builds the detection list directly from the grounded entities
with no deduplication step. `chat()` raises `NotImplementedError`, because
Kosmos-2 is driven by a `<grounding>` prompt rather than a chat template.
LibreYOLO's CLI does not cover this tier: there is no
`libreyolo predict model=...` form for it. See [prediction](/docs/predict) for
sources, streaming and result handling.

## Variants

One size: `kosmos-2-patch14-224`, at 224 px, loaded as `LibreVLM("kosmos-2")`.
It is a 2023-era model, and LibreYOLO's own wrapper notes its grounding is
coarser than the newer detectors in this tier.

LibreYOLO does not train, validate or export Kosmos-2: `train()`, `val()` and
`export()` all raise `NotImplementedError` for every family in this tier (see
the support tier above). Fine-tune Kosmos-2 upstream and load the resulting
weights if you need a custom vocabulary baked in; check `predict()` output by
eye instead of a COCO-style validation pass, since every detection carries the
same placeholder confidence.

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
