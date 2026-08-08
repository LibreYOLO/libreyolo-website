---
title: OV-DEIM
families: [ov_deim]
seo_title: "OV-DEIM in LibreYOLO: open-vocabulary detection"
description: "Use OV-DEIM in LibreYOLO for real-time, DETR-style open-vocabulary detection. Install the openvocab extra and predict with a free-text vocabulary."
lead: "OV-DEIM is a DETR-style open-vocabulary object detector that matches decoder queries to text embeddings from a bundled MobileCLIP text tower. LibreYOLO ports it natively as a predict-only family in its open-vocabulary detector tier."
keywords: [OV-DEIM, DEIMv2, open-vocabulary object detection, real-time detection, zero-shot detection, LibreOpenVocab]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Replace the vocabulary
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # A second call to set_classes() replaces the vocabulary outright and
        # re-embeds it through the text tower; an empty result is a valid
        # outcome rather than an error.
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
---

## Install

OV-DEIM loads through LibreYOLO's open-vocabulary detector tier, which needs
the `openvocab` extra:

```bash
pip install "libreyolo[openvocab]"
```

Unlike the rest of this tier, OV-DEIM is a native LibreYOLO port rather than a
`transformers` wrapper, no `transformers` model class exists for it, but the
same extra covers the `huggingface_hub`, `safetensors`, `regex` and `ftfy`
packages it needs at predict time.

## Predict

OV-DEIM is not a checkpoint LibreYOLO loads through `LibreYOLO()`. It loads
through the sibling `LibreOpenVocab` factory, which downloads a Hugging Face
snapshot on first use and caches it under `weights/`.

<code-tabs name="predict" />

`set_classes()` sets a sticky text vocabulary: call it again to replace the
list outright, or skip it to keep the default COCO-80 labels, and an empty
result is a valid outcome rather than an error. Each decoder query is scored
by cosine similarity against text embeddings from a bundled MobileCLIP-B(LT)
text tower, computed online for whatever vocabulary is set and cached until it
changes, so arbitrary prompts work without any precomputed embedding file.

OV-DEIM has no text-token threshold: only `conf` filters detections, and
passing `text_threshold` raises. Matching is one-to-one top-K selection, so
nothing here runs non-maximum suppression, and `iou` is accepted for API
compatibility but warns and does nothing. `imgsz` and `augment=True` are
rejected outright: the model owns a fixed letterboxed input, and test-time
augmentation is out of scope for this tier. `predict()` on a single image
returns one `Results`, not a list; pass a directory, a list of images, or
`stream=True` for a video source to get several. There is no CLI path for this
family, `libreyolo predict` only loads `.pt` checkpoints through
`LibreYOLO()`, so `LibreOpenVocab` families run from Python. See
[prediction](/docs/predict) for source types and streaming.

Every call to `predict()` also runs the bundled MobileCLIP-B(LT) text tower to
embed the current vocabulary; see Licensing for what that adds to the terms.

## Variants

Three checkpoints, `s`, `m` and `l`. `s` is this tier's default size when none
is given. Unlike the rest of this tier, OV-DEIM is a native port rather than a
`transformers` wrapper: LibreYOLO vendors the detector modules under the same
Apache-2.0 license as the upstream code and reuses the DINOv3 backbone adapter
already built for the DEIMv2 family. The `l` checkpoint's backbone is a
DINOv3-S fine-tune, licensed separately under Meta's DINOv3 License. No
accuracy or latency numbers are published for this family yet.

Training, dataset validation and export are all out of scope for this tier:
`train()`, `val()` and `export()` all raise `NotImplementedError`
unconditionally. This is a predict-only wrapper around a published checkpoint.

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

OV-DEIM layers three upstream licenses onto every prediction call: the
detector weights under OV-DEIM's own CC BY-NC 4.0, the online text tower
under Apple's Machine Learning Research Model license (research use only),
and, for the `l` checkpoint, a DINOv3-S backbone fine-tune under Meta's
DINOv3 License. All three license texts ship inside the LibreYOLO weight
repository.

</provenance-box>

## Citation

<citation-block />
