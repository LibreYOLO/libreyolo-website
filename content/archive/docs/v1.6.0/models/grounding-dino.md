---
title: Grounding DINO
families: [grounding_dino]
seo_title: "Grounding DINO in LibreYOLO: open-set detection"
description: "Use Grounding DINO in LibreYOLO to detect any text-described object. Install the openvocab extra and predict with a free-text vocabulary."
lead: "Grounding DINO is an open-set object detector, developed by IDEA Research, that scores an image against a free-text prompt instead of a fixed class list. LibreYOLO wraps it as a predict-only family in its open-vocabulary detector tier."
keywords: [Grounding DINO, open-vocabulary object detection, open-set detection, zero-shot detection, text-conditioned detector, LibreOpenVocab]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Text threshold
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filters by box score, text_threshold by the decoded phrase's
        # token score. Both default to 0.25 when left unset.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
        print(result.names)
---

## Install

Grounding DINO loads through LibreYOLO's open-vocabulary detector tier, which
needs the `openvocab` extra:

```bash
pip install "libreyolo[openvocab]"
```

That extra pulls in `transformers` and `timm`, the Hugging Face libraries this
tier calls into.

## Predict

Grounding DINO is not a checkpoint LibreYOLO loads through `LibreYOLO()`. It
loads through the sibling `LibreOpenVocab` factory, which downloads a Hugging
Face snapshot on first use and caches it under `weights/`.

<code-tabs name="predict" />

`set_classes()` sets a sticky text vocabulary: call it again to replace the
list, or skip it to keep the default COCO-80 labels. Grounding DINO decodes
free-form phrases from its own text output and maps them back to that
vocabulary itself, an exact normalized match wins, a whole-token match is
accepted, and an ambiguous or unmatched phrase is dropped rather than guessed
at, so `school bus` never gets mapped to `bus` or `school` alone. A vocabulary
long enough to exceed the text encoder's token limit is split into several
prompts, run as separate forward passes, and merged back into one set of
detections capped by `max_det`.

`iou` is accepted for API compatibility but warns and does nothing, since
nothing here runs non-maximum suppression. `imgsz` and `augment=True` are
rejected outright: the `transformers` processor owns resizing, and test-time
augmentation is out of scope for this tier. `predict()` on a single image
returns one `Results`, not a list; pass a directory, a list of images, or
`stream=True` for a video source to get several. There is no CLI path for this
family, `libreyolo predict` only loads `.pt` checkpoints through
`LibreYOLO()`, so `LibreOpenVocab` families run from Python. See
[prediction](/docs/predict) for source types and streaming.

## Variants

Two checkpoints, `t` and `b`. `t` is this tier's default size when none is
given. Both mirror the official IDEA Research release through `transformers`'
`GroundingDinoForObjectDetection`, downloaded once into a LibreYOLO-hosted
Hugging Face snapshot that preserves the upstream files. No accuracy or
latency numbers are published for this family yet.

Training, dataset validation and export are all out of scope for this tier:
`train()`, `val()` and `export()` all raise `NotImplementedError`
unconditionally. This is a predict-only wrapper around a published checkpoint.

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box></provenance-box>

## Citation

<citation-block />
