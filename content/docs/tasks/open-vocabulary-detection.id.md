---
title: Open-vocabulary detection
seo_title: Open-vocabulary detection in LibreYOLO
description: >-
  Detect objects from a text vocabulary in LibreYOLO. Load Grounding DINO,
  OWLv2, OMDet-Turbo or OV-DEIM through LibreOpenVocab and set classes at
  runtime.
lead: >-
  Open-vocabulary detection replaces a checkpoint's fixed class list with words
  you choose at call time. In LibreYOLO it is not a separate task: it is the
  detect task served by a separate model tier, loaded through the LibreOpenVocab
  factory instead of LibreYOLO.
keywords:
  - open vocabulary detection
  - zero shot object detection
  - open set detection
  - grounding dino python
  - owlv2
  - omdet turbo
  - text prompt detection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Swap the vocabulary
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")

        # set_classes is sticky: it holds until the next call to it.
        # Labels must be unique once lowercased and stripped of articles.
        model.set_classes(["a red backpack", "traffic cone"])
        result = model.predict(SAMPLE_IMAGE)

        model.set_classes(["bicycle wheel"])
        result = model.predict(SAMPLE_IMAGE)
    - label: Grounding DINO text threshold
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-b")
        model.set_classes(["remote control", "school bus"])

        # conf filters by box score, text_threshold by the decoded phrase's
        # token score. Both default to 0.25 when left unset. Only Grounding
        # DINO accepts text_threshold; the others raise.
        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## Definition

Open-vocabulary detection returns ordinary detection `Results`: boxes,
confidences and class indices, with `result.names` mapping those indices back to
the strings you asked for. What changes is where the class list comes from.
A conventional detector is trained against a fixed set of categories and can
never emit a category outside it. These models take the vocabulary as text at
inference time, so `set_classes(["forklift", "safety cone"])` is enough to make
those the classes.

LibreYOLO has no `open-vocabulary` task key. These models declare
`SUPPORTED_TASKS = ("detect",)` like any other detector. What separates them is
the loading path: they are Hugging Face snapshots rather than LibreYOLO
state-dict checkpoints, so they stay out of the `LibreYOLO()` factory and are
constructed through `LibreOpenVocab()` instead. That factory is a sibling of
`LibreSAM()` and `LibreVLM()`, not a replacement for `LibreYOLO()`.

Scores are real detection scores, not a generated caption parsed after the fact.
Each family scores image regions against the text embedding of every prompt.

## Models

Four families make up the tier, all of them predict only. Load any of them by
alias through `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino), from IDEA Research, in `t` and
`b` sizes. It is the tier default, and the only family that accepts
`text_threshold`, a second cutoff on the decoded phrase's token score.

[OWLv2](/docs/models/owlv2), from Google Research, in `b16` and `l14` sizes.
It scores image regions against text embeddings from a CLIP-style encoder.

[OMDet-Turbo](/docs/models/omdet-turbo), from Om AI Lab, in one `t` size. It
decouples class embeddings from a language task prompt, and is the one family
here that suppresses overlapping boxes inside its own post-processing, so `iou=`
is honored.

[OV-DEIM](/docs/models/ov-deim), in `s`, `m` and `l` sizes, a DETR-style
detector that matches decoder queries to text embeddings from a bundled
MobileCLIP text tower. It is one-to-one matching with top-K selection, so no NMS
runs anywhere.

OV-DEIM's weights are the restricted case in this tier. The detector weights are
CC BY-NC 4.0, non-commercial. The bundled text tower is under Apple's Machine
Learning Research Model license, research use only. The `l` checkpoint adds a
DINOv3-S backbone fine-tune under Meta's DINOv3 License. All three license texts
ship inside the weight repository, and the library logs the same summary when it
resolves the weights, before the model is built. Read
[OV-DEIM](/docs/models/ov-deim) before deploying it.

The tier needs one extra:

```bash
pip install "libreyolo[openvocab]"
```

That covers `transformers` and `timm` for the three wrapped families, and the
`huggingface_hub`, `safetensors`, `regex` and `ftfy` packages OV-DEIM needs as a
native port.

A second tier also takes a text vocabulary: `LibreVLM()` loads generative
vision-language models, such as [Qwen3-VL](/docs/models/qwen3-vl) and
[Florence-2](/docs/models/florence-2), and turns their output into the same
`Results`. It shares the `set_classes()` surface. The difference is what
produces the boxes: the families on this page are discriminative detectors that
emit scores directly, while the VLM tier generates them.

## Predict

<code-tabs name="predict" />

`set_classes()` takes a non-empty list of label strings and holds until it is
called again. Labels must be unique once lowercased and stripped of leading
articles, so `"a bus"` and `"bus"` cannot coexist in one vocabulary. Multi-word
phrases are labels like any other, and each family turns the list into its own
text input before tokenizing, so `"traffic cone"` is a different query from
`"cone"`.

Three prediction arguments behave differently here than on a native detector.
`imgsz=` is rejected, because the processor owns resizing for these families.
`augment=True` is rejected, since test-time augmentation is out of scope for the
tier. `iou=` applies only to the family whose processor runs its own
suppression; where nothing is suppressed, passing it warns and is ignored.

Left unset, `conf` takes the loaded family's own default rather than
`predict()`'s usual 0.25, and that default is not the same across the tier. Set
it explicitly when comparing two families on the same image.

`track()` raises across the tier. Run `predict()` per frame instead. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Train

No family in this tier trains inside LibreYOLO. `train()` raises: fine-tune
upstream and load the resulting weights. The vocabulary passed to
`set_classes()` is the only setting that changes what a loaded model detects.

## Validate

There is no validator for this tier, and `val()` raises. Open-vocabulary
validation needs a dedicated one, because the standard detection validator feeds
image tensors straight to the model, while these families require
text-conditioned inputs built alongside them.

## Export

Export is out of scope for the tier and `export()` raises. These models run
through `predict()` in PyTorch.


