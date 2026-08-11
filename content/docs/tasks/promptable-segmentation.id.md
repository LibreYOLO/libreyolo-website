---
title: Promptable segmentation
seo_title: Promptable segmentation in LibreYOLO
description: >-
  Turn a point, box or text concept into an object mask in LibreYOLO. Load SAM,
  SAM 2, SAM 3, EdgeTAM, MobileSAM or PicoSAM3 through LibreSAM.
lead: >-
  Promptable segmentation turns a click into a mask: you point at an object, or
  draw a box around it, and the model returns its outline. In LibreYOLO it is
  not a separate task key but a model tier, loaded through the LibreSAM factory,
  whose results are ordinary segmentation Results.
keywords:
  - promptable segmentation
  - interactive segmentation
  - segment anything python
  - point prompt
  - box prompt
  - SAM python
  - mask from click
last_verified: 1.5.0
snippets:
  predict:
    - label: Point and box prompts
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # A point is [x, y] in pixels; labels are 1 positive, 0 negative.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # polygons
        print(result.boxes.xyxy)    # tight boxes derived from the masks

        # A box prompt gives one mask per box.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Encode once, prompt many times'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # set_image runs the heavy image encoder once and caches it.
        model.set_image(SAMPLE_IMAGE)
        first = model.predict(points=[640, 420], labels=[1])
        second = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
    - label: Segment everything
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # No prompt means a grid of points over the whole image. The default
        # grid of 32 per side is ~1024 decoder passes, which is slow on CPU.
        result = model.predict(SAMPLE_IMAGE, points_per_side=8)
        print(len(result.masks))
    - label: Ambiguity masks
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # One point can mean a sleeve, a shirt, or a person. multimask=True
        # returns all three whole-versus-part masks instead of the best one.
        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )
        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Definition

Promptable segmentation takes an image plus a spatial prompt and returns the
mask of whatever the prompt points at. Nothing is classified: there is no class
list, and `result.boxes` holds tight boxes derived from the masks rather than
detections in their own right. `result.masks` carries the mask data and
`result.masks.xy` its polygons.

The prompt is the interface. `points` is `[x, y]` pixel coordinates, one set per
object, with `labels` marking each point positive (1, include this) or negative
(0, exclude this). `bboxes` is `[x1, y1, x2, y2]`, one mask per box. Points and
boxes can be combined, in which case they pair per object and must be the same
length. Omitting every prompt runs the segment-everything path, a grid of points
over the image.

A single point is ambiguous by construction. Clicking a sleeve could mean the
sleeve, the shirt or the person, so `multimask=True` returns all three
whole-versus-part masks per prompt instead of the single best one. `conf`
filters on the model's predicted IoU, a mask-quality score, not a detection
confidence.

LibreYOLO has no `promptable` task key. The tier registers as `segment`, the
same key instance segmentation uses. What separates it is the call shape, which
is why it has its own factory, `LibreSAM()`, a sibling of `LibreYOLO()`,
`LibreOpenVocab()` and `LibreVLM()`. A single `predict(image)` signature cannot
express the loop these models are built for: `set_image()` runs the image
encoder once and caches the embeddings, every later `predict()` call with
`source=None` pays only for prompt decoding, and `reset_image()` clears the
cache. The image encoder is the dominant cost and runs once per image, so a
second prompt on the same image skips it entirely.

## Models

Six families load through `LibreSAM` by alias.

[SAM](/docs/models/sam) is the default, in `base`, `large` and `huge` sizes,
also spelled `b`, `l` and `h`.

[SAM 2](/docs/models/sam-2), as `sam2-tiny`, `sam2-small`, `sam2-base-plus` and
`sam2-large`. LibreYOLO supports its image path.

[SAM 3](/docs/models/sam-3), as `sam3`, is the one family that accepts a text
concept prompt: `text="yellow school bus"` returns every matching instance.
Passing `text=` to any other family raises with a message naming SAM 3. Its
weights come from Meta under the custom SAM License rather than LibreYOLO's MIT
license, and the repository is gated: accept the terms on the model page and
authenticate with `hf auth login` before the first download. Read
[SAM 3](/docs/models/sam-3) before deploying it.

[EdgeTAM](/docs/models/edgetam), as `edgetam`, is an on-device variant of SAM 2.
LibreYOLO supports its image path.

[MobileSAM](/docs/models/mobilesam), as `mobilesam`, replaces SAM's ViT-H
encoder with a distilled TinyViT one.

[PicoSAM3](/docs/models/picosam3), as `picosam3`, is a compact CNN for
box-prompted regions on edge sensors. Box prompts are the whole contract here:
points, text, mask, multimask and segment-everything all raise with a message
pointing at SAM 2 or SAM 3.

The tier's extra covers the four families that load through `transformers`:

```bash
pip install "libreyolo[sam]"
```

MobileSAM and PicoSAM3 are native LibreYOLO ports and need no `transformers`
install to run.

## Predict

<code-tabs name="predict" />

`source` and `set_image()` are alternatives, not a sequence: pass an image to
`predict()` for a one-shot call, or call `set_image()` first and then
`predict(source=None)` for each prompt. Passing `device=` to `predict()` moves
the model for that call and every later one, and invalidates any cached
embeddings.

Segment-everything is the expensive mode. `points_per_side` defaults to 32,
which is roughly 1024 decoder passes over the image; lower it for anything
interactive on CPU. In that mode `conf` applies the family's grid threshold when
left unset, while in the prompted path an unset `conf` keeps every mask. Pass
`conf=0.0` to disable filtering in either mode, and `max_det` to cap how many
masks come back.

Mask prompts are not supported in this version, and `masks=` raises rather than
being ignored. `track()` also raises across the tier: these are image
segmenters, so run `predict()` per frame. See [prediction](/docs/predict) for
sources and result handling.

## Train

No family in this tier trains inside LibreYOLO. `train()` raises: fine-tune
upstream and load the resulting weights.

## Validate

There is no validator for this tier, and `val()` raises. A promptable mask has
no fixed class set to score against, so the usual detection and segmentation
metrics have nothing to key on. Scoring a prompted mask means comparing it to a
reference mask you supply yourself, against the prompts you care about.

## Export

Export is out of scope for the tier as a whole and `export()` raises, with one
exception. [PicoSAM3](/docs/models/picosam3) exports its raw 96x96 region CNN to
ONNX as `roi_image -> mask_logits`; box cropping and the mask resize back to
image coordinates stay in Python. Every other family runs through `predict()` in
PyTorch. See [export](/docs/export) for the formats available elsewhere in the
library.


