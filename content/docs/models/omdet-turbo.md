---
title: OMDet-Turbo
families: [omdet_turbo]
seo_title: "OMDet-Turbo in LibreYOLO: real-time zero-shot detection"
description: "Use OMDet-Turbo in LibreYOLO for real-time open-vocabulary detection. Install the openvocab extra and predict with a free-text vocabulary."
lead: "OMDet-Turbo is a real-time open-vocabulary object detector, developed by Om AI Lab, that decouples class embeddings from a language task prompt. LibreYOLO wraps it as a predict-only family in its open-vocabulary detector tier."
keywords: [OMDet-Turbo, OmDet, open-vocabulary object detection, real-time detection, zero-shot detection, LibreOpenVocab]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Custom NMS threshold
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["traffic light", "bicycle"])

        # OMDet-Turbo is the one family in this tier that honours iou=: its
        # own post-processing takes the suppression threshold as an argument,
        # defaulting to 0.5 when iou= is left unset.
        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)
        print(result.names, len(result))
---

## Install

OMDet-Turbo loads through LibreYOLO's open-vocabulary detector tier, which
needs the `openvocab` extra:

```bash
pip install "libreyolo[openvocab]"
```

That extra pulls in `transformers` and `timm`, the Hugging Face libraries this
tier calls into; OMDet-Turbo's Swin backbone loads through `transformers`'
`TimmBackbone` wrapper.

## Predict

OMDet-Turbo is not a checkpoint LibreYOLO loads through `LibreYOLO()`. It loads
through the sibling `LibreOpenVocab` factory, which downloads a Hugging Face
snapshot on first use and caches it under `weights/`.

<code-tabs name="predict" />

`set_classes()` sets a sticky text vocabulary: call it again to replace the
list outright, or skip it to keep the default COCO-80 labels, and an empty
result is a valid outcome rather than an error. Unlike Grounding DINO,
OMDet-Turbo decouples its class embeddings from the language task prompt, so
`transformers`' post-processing returns labels that map straight back to the
queried class list with no phrase-disambiguation step.

OMDet-Turbo has no text-token threshold: only `conf` filters detections, and
passing `text_threshold` raises. It is the one family in this tier that runs
its own non-maximum suppression inside
`post_process_grounded_object_detection`, so `iou` is honoured here rather
than warned about. `imgsz` and `augment=True` are rejected outright: the
`transformers` processor owns resizing, and test-time augmentation is out of
scope for this tier. `predict()` on a single image returns one `Results`, not
a list; pass a directory, a list of images, or `stream=True` for a video
source to get several. There is no CLI path for this family, `libreyolo
predict` only loads `.pt` checkpoints through `LibreYOLO()`, so
`LibreOpenVocab` families run from Python. See [prediction](/docs/predict) for
source types and streaming.

## Variants

One checkpoint, `t`, the tier's only size. It mirrors `omlab/omdet-turbo-swin-tiny-hf`
at a pinned upstream revision through `transformers`'
`OmDetTurboForObjectDetection`; the mirrored weight file is byte-identical to
that upstream snapshot. No accuracy or latency numbers are published for this
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
