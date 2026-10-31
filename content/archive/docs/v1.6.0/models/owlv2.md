---
title: OWLv2
families: [owlv2]
seo_title: "OWLv2 in LibreYOLO: zero-shot object detection"
description: "Use OWLv2 in LibreYOLO to detect any text-described object. Install the openvocab extra and predict with a free-text vocabulary."
lead: "OWLv2 is an open-vocabulary object detector, developed by Google Research, that scores image regions against text embeddings from a CLIP-style encoder. LibreYOLO wraps it as a predict-only family in its open-vocabulary detector tier."
keywords: [OWLv2, OWL-ViT, open-vocabulary object detection, zero-shot detection, text-conditioned detector, LibreOpenVocab]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Default vocabulary
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        # Skipping set_classes() keeps the tier's default COCO-80 vocabulary.
        model = LibreOpenVocab("owlv2-l14")
        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        print(result.names)
---

## Install

OWLv2 loads through LibreYOLO's open-vocabulary detector tier, which needs the
`openvocab` extra:

```bash
pip install "libreyolo[openvocab]"
```

That extra pulls in `transformers` and `timm`, the Hugging Face libraries this
tier calls into.

## Predict

OWLv2 is not a checkpoint LibreYOLO loads through `LibreYOLO()`. It loads
through the sibling `LibreOpenVocab` factory, which downloads a Hugging Face
snapshot on first use and caches it under `weights/`.

<code-tabs name="predict" />

`set_classes()` sets a sticky text vocabulary: call it again to replace the
list, or skip it to keep the default COCO-80 labels. Each label is wrapped in
a fixed prompt template before it reaches the text tower, matching how
`transformers`' `Owlv2ForObjectDetection` was trained.

OWLv2 has no text-token threshold: only `conf` filters detections, and passing
`text_threshold` raises. `iou` is accepted for API compatibility but warns and
does nothing, since nothing here runs non-maximum suppression. `imgsz` and
`augment=True` are rejected outright: the `transformers` processor owns
resizing, and test-time augmentation is out of scope for this tier.
`predict()` on a single image returns one `Results`, not a list; pass a
directory, a list of images, or `stream=True` for a video source to get
several. There is no CLI path for this family, `libreyolo predict` only loads
`.pt` checkpoints through `LibreYOLO()`, so `LibreOpenVocab` families run from
Python. See [prediction](/docs/predict) for source types and streaming.

## Variants

Two checkpoints, `b16` (base, patch size 16) and `l14` (large, patch size 14).
`b16` is this tier's default size when none is given. Both mirror the official
Google Research release through `transformers`' `Owlv2ForObjectDetection`,
downloaded once into a LibreYOLO-hosted Hugging Face snapshot that preserves
the upstream files. No accuracy or latency numbers are published for this
family yet.

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
